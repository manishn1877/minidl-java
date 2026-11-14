package minidl;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;

/**
 * Handwriting recognition web app, in pure Java (only the JDK's built-in web server, no libraries).
 *
 *   java -cp out minidl.HandwritingServer [auto|model.bin] [port=8080] [data-folder]
 *
 * With "auto" (the default) the server loads EVERY trained model it finds in the current folder and the page lets you
 * switch between them:
 *   alnum-cnn.bin          digits AND letters (36 classes)      key "all"
 *   emnist-letters-cnn.bin letters A-Z (26 classes)             key "letters"
 *   mnist-cnn-aug.bin      digits 0-9 (10 classes)              (or mnist-cnn.bin)   key "digits"
 * The page opens with the broadest model, and says clearly which models are missing: a digits-only model can only
 * answer with digits, so it reads a handwritten "B" as 8 or 6. Nothing is silently substituted.
 * data-folder (default ".") holds mnist/ and emnist/ : it is used for the test accuracy and for mixing real pictures
 * into corrections. With a model file instead of "auto", only that model is loaded.
 *
 * API:  GET /api/info    POST /api/predict?w=&amp;h=&amp;mode=single|multi&amp;tta=1&amp;model=all   (body = w*h bytes of ink, 0..255)
 *       POST /api/learn?w=&amp;h=&amp;label=7&amp;model=all   POST /api/save?model=all   POST /api/reset?model=all
 */
public class HandwritingServer {
    /** One loaded model. */
    static final class Slot {
        String key, name, file;
        DigitRecognizer rec;
        volatile double testAcc = -1;
        boolean replayTried;
    }

    static final Map<String, Slot> slots = new LinkedHashMap<>();
    static String defaultKey, warning = "", dataDir = ".";
    static File webDir = new File(System.getProperty("minidl.web", "web"));

    public static void main(String[] args) throws Exception {
        String which = args.length > 0 ? args[0] : "auto";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 8080;
        if (args.length > 2) dataDir = args[2];

        if (which.equals("auto")) {
            for (String f : new String[]{"emnist-full-cnn.bin", "emnist-case-cnn.bin", "alnum-cnn.bin", "emnist-letters-cnn.bin", "mnist-cnn-aug.bin"}) if (new File(f).exists()) addSlot(f);
            if (!slots.containsKey("digits") && new File("mnist-cnn.bin").exists()) addSlot("mnist-cnn.bin");
        } else if (new File(which).exists()) {
            addSlot(which);
        } else if (which.contains("emnist") || which.contains("alnum")) {
            System.out.println("model file '" + which + "' not found. Train it first with ./train-all.sh (it needs the EMNIST data: ./get-emnist.sh).");
            return;
        } else if (new File(dataDir, "mnist").isDirectory() || loadSplit(10, dataDir, true) != null) { // asked for a digits model that does not exist yet: train a quick one
            System.out.println("no " + which + " yet: training a CNN for " + System.getProperty("minidl.epochs", "3") + " epochs first...");
            ByteSource all = loadSplit(10, dataDir, true);
            Sequential m = train(all);
            ModelIO.saveBinary(m, which);
            addSlot(which);
        }
        if (slots.isEmpty()) {
            System.out.println("No trained model found in this folder. Train one first:\n"
                + "  digits 0-9:            java -cp out minidl.MnistDemo mnist 8 cnn 60000 aug      (makes mnist-cnn-aug.bin)\n"
                + "  everything at once:    ./train-all.sh                                           (digits, letters, and digits+letters)");
            return;
        }
        defaultKey = slots.keySet().iterator().next();
        warning = buildWarning();
        if (!warning.isEmpty()) System.out.println("WARNING: " + warning);
        if (!new File(webDir, "index.html").exists() && HandwritingServer.class.getResource("/web/index.html") == null) System.out.println("warning: " + new File(webDir, "index.html").getAbsolutePath() + " not found (run from the project folder)");

        Thread acc = new Thread(HandwritingServer::measureAccuracies, "test-accuracy");
        acc.setDaemon(true);
        acc.start();

        startHttp(port);
        StringBuilder names = new StringBuilder();
        for (Slot s : slots.values()) names.append(names.length() > 0 ? ", " : "").append(s.key).append(" = ").append(s.file);
        System.out.println("models loaded: " + names + "   (page opens with '" + defaultKey + "')");
        System.out.println("open http://localhost:" + port + "   (Ctrl+C to stop)");
    }

    /** Starts the web server (localhost only) and returns it. */
    static HttpServer startHttp(int port) throws IOException {
        HttpServer http = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        http.createContext("/", ex -> guard(ex, HandwritingServer::page));
        http.createContext("/api/info", ex -> guard(ex, HandwritingServer::info));
        http.createContext("/api/predict", ex -> guard(ex, HandwritingServer::predict));
        http.createContext("/api/learn", ex -> guard(ex, HandwritingServer::learn));
        http.createContext("/api/save", ex -> guard(ex, HandwritingServer::save));
        http.createContext("/api/reset", ex -> guard(ex, HandwritingServer::reset));
        http.setExecutor(Executors.newFixedThreadPool(4));
        http.start();
        return http;
    }

    static void addSlot(String file) throws IOException {
        Sequential model = ModelIO.loadModelBinary(file, new Random(1));
        Slot s = new Slot();
        s.rec = new DigitRecognizer(model);
        s.file = file;
        s.key = s.rec.classes == 62 ? "full" : s.rec.classes == 52 ? "case" : s.rec.classes == 36 ? "all" : s.rec.classes == 26 ? "letters" : "digits";
        s.name = s.rec.classes == 62 ? "digits + letters (A-Z a-z)" : s.rec.classes == 52 ? "letters (A-Z a-z)" : s.rec.classes == 36 ? "digits + letters" : s.rec.classes == 26 ? "letters A-Z" : "digits 0-9";
        slots.put(s.key, s);
    }

    static String buildWarning() {
        boolean cased = slots.containsKey("full") || slots.containsKey("case");
        if (slots.containsKey("all") || slots.containsKey("letters") || cased) {
            return cased ? "" : "No model with capital AND small letters yet (a c and a C look the same to the merged A-Z model). Run ./train-all.sh to train the A-Z a-z model, then restart.";
        }
        if (!slots.containsKey("letters"))
            return "Only the digits model is loaded (" + slots.get("digits").file + "), so everything you write is read as a digit: a letter becomes 8, 6, 0 ... "
                + "Run ./train-all.sh (needs the data), then restart ./run-web.sh.";
        return "The digits + letters model (alnum-cnn.bin) is not trained yet. Use the letters or digits model for now, or run ./train-all.sh then restart.";
    }

    /** Test accuracy of every model on its own unseen test pictures (needs the data folder); runs in the background after the start. */
    static void measureAccuracies() {
        for (Slot s : slots.values()) {
            try {
                ByteSource t = loadSplit(s.rec.classes, dataDir, false);
                if (t == null) continue;
                s.testAcc = s.rec.accuracy(t);
                System.out.printf("model %s (%s): %.2f%% on %d unseen test pictures%n", s.file, s.name, s.testAcc, t.size());
            } catch (Exception | OutOfMemoryError e) { System.out.println("could not measure " + s.file + ": " + e); }
        }
    }

    /** Real pictures to mix into corrections (so the model does not forget); loaded the first time a correction is made. */
    static void ensureReplay(Slot s) {
        synchronized (s) {
            if (s.replayTried) return;
            s.replayTried = true;
            try {
                ByteSource tr = loadSplit(s.rec.classes, dataDir, true);
                if (tr != null) s.rec.replayFrom(tr.view(0, tr.size() - tr.size() / 10));
            } catch (Exception | OutOfMemoryError e) { System.out.println("no replay data for " + s.file + ": " + e); }
        }
    }

    /** The train or test pictures that match a model with this many classes, or null when the data folder does not have them. */
    static ByteSource loadSplit(int classes, String dir, boolean train) throws IOException {
        if (dir == null) return null;
        String split = train ? "train" : "test", pre = train ? "train" : "t10k";
        int max = train ? 60000 : Integer.MAX_VALUE;
        if (classes == 62 || classes == 52) {
            String e = EmnistLoader.existsByClass(dir) ? dir : new File(dir, "emnist").getPath();
            return EmnistLoader.existsByClass(e) ? EmnistLoader.byClass(e, split, train ? 60000 : 20000, classes == 52) : null;
        }
        if (classes == 36) {
            String m = new File(dir, "mnist").getPath(), e = new File(dir, "emnist").getPath();
            return Alnum.exists(m, e) ? Alnum.load(m, e, split, max) : null;
        }
        if (classes == 26) {
            String e = EmnistLoader.exists(dir) ? dir : new File(dir, "emnist").getPath();
            return EmnistLoader.exists(e) ? EmnistLoader.lettersFrom(e, split, Integer.MAX_VALUE) : null;
        }
        String m = new File(find(dir, pre + "-images-idx3-ubyte")).exists() ? dir : new File(dir, "mnist").getPath();
        if (!new File(find(m, pre + "-images-idx3-ubyte")).exists()) return null;
        return MnistLoader.loadSource(find(m, pre + "-images-idx3-ubyte"), find(m, pre + "-labels-idx1-ubyte"), max);
    }

    // ------------------------------------------------------------------ handlers
    interface Handler { void run(HttpExchange ex) throws Exception; }

    /** Websites that may talk to this server from a browser (default: the project's GitHub Pages site). Change with -Dminidl.cors=https://a.example,https://b.example */
    static final java.util.Set<String> ALLOWED_ORIGINS = new java.util.HashSet<>(java.util.Arrays.asList(System.getProperty("minidl.cors", "https://manishn1877.github.io").split(",")));

    /**
     * Cross-origin rules. A page from an allowed origin may call the API (needed when the page is hosted on GitHub Pages and the
     * Java server runs on your own computer). Every POST must carry the header X-MiniDL: a browser then asks permission first
     * (a preflight), and only an allowed origin gets it, so a random website cannot make your browser change your local model.
     * Returns false when the request is already answered.
     */
    static boolean cors(HttpExchange ex) throws IOException {
        String origin = ex.getRequestHeaders().getFirst("Origin");
        boolean allowed = origin != null && ALLOWED_ORIGINS.contains(origin);
        if (allowed) {
            ex.getResponseHeaders().set("Access-Control-Allow-Origin", origin);
            ex.getResponseHeaders().set("Vary", "Origin");
            ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, X-MiniDL");
            ex.getResponseHeaders().set("Access-Control-Allow-Private-Network", "true"); // an https page calling localhost
            ex.getResponseHeaders().set("Access-Control-Max-Age", "600");
        }
        if (ex.getRequestMethod().equals("OPTIONS")) { ex.sendResponseHeaders(allowed ? 204 : 403, -1); ex.close(); return false; }
        if (ex.getRequestMethod().equals("POST") && ex.getRequestHeaders().getFirst("X-MiniDL") == null) {
            send(ex, 403, "application/json", "{\"error\":\"missing header X-MiniDL (a safety rule: other websites must not be able to change your model)\"}");
            return false;
        }
        return true;
    }

    static void guard(HttpExchange ex, Handler h) throws IOException {
        try { if (!cors(ex)) return; h.run(ex); }
        catch (IllegalArgumentException e) { send(ex, 400, "application/json", "{\"error\":" + str(e.getMessage()) + "}"); }
        catch (Exception e) { send(ex, 500, "application/json", "{\"error\":" + str(String.valueOf(e)) + "}"); }
    }

    static Slot slot(Map<String, String> q) {
        String k = q.getOrDefault("model", defaultKey);
        Slot s = slots.get(k);
        if (s == null) throw new IllegalArgumentException("model '" + k + "' is not loaded (loaded: " + slots.keySet() + ")");
        return s;
    }

    static void page(HttpExchange ex) throws IOException {
        String p = ex.getRequestURI().getPath();
        if (!p.equals("/") && !p.equals("/index.html")) { send(ex, 404, "text/plain", "not found"); return; }
        File f = new File(webDir, "index.html");
        if (f.exists()) { send(ex, 200, "text/html; charset=utf-8", new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8)); return; }
        try (java.io.InputStream in = HandwritingServer.class.getResourceAsStream("/web/index.html")) { // the copy packed inside minidl.jar
            if (in == null) { send(ex, 404, "text/plain", "web/index.html not found"); return; }
            send(ex, 200, "text/html; charset=utf-8", new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    static void info(HttpExchange ex) throws IOException {
        StringBuilder sb = new StringBuilder("{\"default\":" + str(defaultKey) + ",\"warning\":" + str(warning) + ",\"models\":[");
        boolean first = true;
        for (Slot s : slots.values()) {
            if (!first) sb.append(',');
            first = false;
            sb.append("{\"key\":").append(str(s.key)).append(",\"name\":").append(str(s.name)).append(",\"file\":").append(str(new File(s.file).getName()))
              .append(",\"alphabet\":").append(str(s.rec.alphabet)).append(",\"params\":").append(s.rec.paramCount())
              .append(",\"kb\":").append(String.format("%.1f", s.rec.weightBytes() / 1024.0)).append(",\"learned\":").append(s.rec.learnedCount())
              .append(",\"testAcc\":").append(s.testAcc).append('}');
        }
        send(ex, 200, "application/json", sb.append("]}").toString());
    }

    static void predict(HttpExchange ex) throws Exception {
        Map<String, String> q = query(ex);
        Slot s = slot(q);
        int[] wh = new int[2];
        float[] ink = readInk(ex, q, wh);
        boolean multi = "multi".equals(q.get("mode")), tta = !"0".equals(q.get("tta"));
        String caseMode = q.getOrDefault("case", "auto");
        double refH = 0;
        try { refH = Double.parseDouble(q.getOrDefault("ref", "0")); } catch (NumberFormatException e) { throw new IllegalArgumentException("ref must be a number"); }
        long t0 = System.nanoTime();
        List<DigitPreprocess.Piece> pieces = new ArrayList<>();
        if (multi) pieces = DigitPreprocess.segment(ink, wh[0], wh[1], s.rec.style(), px -> s.rec.predict(px, false).confidence);
        else { DigitPreprocess.Piece p = DigitPreprocess.whole(ink, wh[0], wh[1], s.rec.style()); if (p != null) pieces.add(p); }
        StringBuilder digits = new StringBuilder(), text = new StringBuilder();
        for (DigitPreprocess.Piece p : pieces) {
            DigitRecognizer.Result r = s.rec.predict(p.pixels, tta);
            LetterCase.Outcome oc = LetterCase.apply(s.rec.alphabet, r.probs, caseMode, p.h, refH); // capital or small, decided by size for look-alike letters
            if (oc.note != null || oc.amb) r = new DigitRecognizer.Result(oc.probs);
            if (digits.length() > 0) digits.append(',');
            digits.append(digitJson(s, p, r, oc.amb, oc.note));
            text.append(r.confidence < 0.5 ? '?' : s.rec.alphabet.charAt(r.label));
        }
        double ms = (System.nanoTime() - t0) / 1e6;
        send(ex, 200, "application/json", String.format("{\"ms\":%.1f,\"model\":%s,\"text\":\"%s\",\"digits\":[%s]}", ms, str(s.key), text, digits));
    }

    static void learn(HttpExchange ex) throws Exception {
        Map<String, String> q = query(ex);
        Slot s = slot(q);
        int[] wh = new int[2];
        float[] ink = readInk(ex, q, wh);
        int label = Integer.parseInt(q.getOrDefault("label", "-1"));
        DigitPreprocess.Piece p = DigitPreprocess.whole(ink, wh[0], wh[1], s.rec.style());
        if (p == null) throw new IllegalArgumentException("nothing drawn");
        ensureReplay(s);
        double loss = s.rec.learn(p.pixels, label);
        try (FileOutputStream f = new FileOutputStream(s.key.equals("digits") ? "feedback.bin" : "feedback-" + s.key + ".bin", true)) { // every correction is kept: 784 pixel bytes + 1 label byte
            byte[] rec1 = new byte[DigitRecognizer.DIM + 1];
            for (int i = 0; i < DigitRecognizer.DIM; i++) rec1[i] = (byte) Math.round(p.pixels[i] * 255);
            rec1[DigitRecognizer.DIM] = (byte) label;
            f.write(rec1);
        }
        send(ex, 200, "application/json", String.format("{\"loss\":%.4f,\"learned\":%d,\"digit\":%s}", loss, s.rec.learnedCount(), digitJson(s, p, s.rec.predict(p.pixels, true), false, null)));
    }

    static void save(HttpExchange ex) throws Exception {
        Slot s = slot(query(ex));
        String out = s.file.replaceAll("\\.bin$", "") + "-tuned.bin";
        s.rec.save(out);
        send(ex, 200, "application/json", "{\"saved\":" + str(out) + "}");
    }

    static void reset(HttpExchange ex) throws IOException {
        slot(query(ex)).rec.reset();
        send(ex, 200, "application/json", "{\"ok\":true}");
    }

    // ------------------------------------------------------------------ helpers
    static float[] readInk(HttpExchange ex, Map<String, String> q, int[] wh) throws IOException {
        int w, h;
        try { w = Integer.parseInt(q.get("w")); h = Integer.parseInt(q.get("h")); } catch (RuntimeException e) { throw new IllegalArgumentException("need w and h"); }
        if (w < 8 || h < 8 || (long) w * h > 4_000_000L) throw new IllegalArgumentException("bad image size");
        byte[] raw = ex.getRequestBody().readNBytes(w * h);
        if (raw.length != w * h) throw new IllegalArgumentException("body must be w*h bytes");
        float[] ink = new float[w * h];
        for (int i = 0; i < ink.length; i++) ink[i] = (raw[i] & 0xFF) / 255f;
        wh[0] = w; wh[1] = h;
        return ink;
    }

    static String digitJson(Slot s, DigitPreprocess.Piece p, DigitRecognizer.Result r) { return digitJson(s, p, r, false, null); }

    static String digitJson(Slot s, DigitPreprocess.Piece p, DigitRecognizer.Result r, boolean amb, String caseNote) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("{\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d,\"label\":%d,\"ch\":\"%c\",\"conf\":%.4f,\"bits\":%.3f,\"amb\":%b,\"note\":%s,\"probs\":[", p.x, p.y, p.w, p.h, r.label, s.rec.alphabet.charAt(r.label), r.confidence, r.entropyBits, amb, p.note != null ? str(p.note) : caseNote != null ? str(caseNote) : "null"));
        for (int i = 0; i < r.probs.length; i++) sb.append(i > 0 ? "," : "").append(String.format("%.4f", r.probs[i]));
        sb.append("],\"view\":[");
        for (int i = 0; i < p.pixels.length; i++) sb.append(i > 0 ? "," : "").append(Math.round(p.pixels[i] * 255));
        return sb.append("]}").toString();
    }

    static Map<String, String> query(HttpExchange ex) {
        Map<String, String> m = new HashMap<>();
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) for (String kv : q.split("&")) { int i = kv.indexOf('='); if (i > 0) m.put(kv.substring(0, i), kv.substring(i + 1)); }
        return m;
    }

    static String str(String s) { return "\"" + String.valueOf(s).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ") + "\""; }

    static void send(HttpExchange ex, int code, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(code, b.length);
        ex.getResponseBody().write(b);
        ex.close();
    }

    static String find(String dir, String name) {
        File gz = new File(dir, name + ".gz");
        return (gz.exists() ? gz : new File(dir, name)).getPath();
    }

    static Sequential train(ByteSource all) {
        int nVal = all.size() / 10, nTrain = all.size() - nVal;
        Sequential m = MnistDemo.cnn(new Random(1));
        new Trainer().epochs(Integer.getInteger("minidl.epochs", 3)).batchSize(64).seed(0).printEvery(1)
            .validation(all.view(nTrain, all.size())).patience(3).schedule(Schedule.cosine(0.002, 0.0001))
            .fit(m, new AugmentedSource(all.view(0, nTrain), 28, 28, 2, 10, 7), new Losses.SoftmaxCrossEntropy(), new Optimizers.Adam(0.002));
        return m;
    }
}
