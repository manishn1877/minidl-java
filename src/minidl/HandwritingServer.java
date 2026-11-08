package minidl;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;

/**
 * Handwriting recognition web app, in pure Java (only the JDK's built-in web server, no libraries).
 *
 *   java -cp out minidl.HandwritingServer [model.bin] [port=8080] [mnist-folder]
 *
 * Open http://localhost:8080 , draw a digit (or a whole number), and the model answers live.
 *   model.bin     a model saved by MnistDemo (default: mnist-cnn-aug.bin, else mnist-cnn.bin)
 *   mnist-folder  optional. If model.bin does not exist yet, the server trains one first (3 epochs).
 *                 With the folder it also reports test accuracy and mixes real digits into corrections.
 *
 * API:  GET /api/info    POST /api/predict?w=&amp;h=&amp;mode=single|multi&amp;tta=1   (body = w*h bytes of ink, 0..255)
 *       POST /api/learn?w=&amp;h=&amp;label=7   POST /api/save   POST /api/reset
 */
public class HandwritingServer {
    static DigitRecognizer rec;
    static String modelPath;
    static File webDir = new File(System.getProperty("minidl.web", "web"));
    static double testAcc = -1;

    public static void main(String[] args) throws Exception {
        modelPath = args.length > 0 ? args[0] : firstExisting("mnist-cnn-aug.bin", "mnist-cnn.bin", "emnist-letters-cnn.bin");
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 8080;
        String mnist = args.length > 2 ? args[2] : null;
        if (modelPath == null) modelPath = "mnist-cnn.bin";

        ByteSource all = null, tst = null;
        Sequential model;
        if (new File(modelPath).exists()) {
            model = ModelIO.loadModelBinary(modelPath, new Random(1));
            int classes = new DigitRecognizer(model).classes;
            if (mnist != null && classes == 26 && EmnistLoader.exists(mnist)) {
                all = EmnistLoader.lettersFrom(mnist, "train", Integer.MAX_VALUE);
                tst = EmnistLoader.lettersFrom(mnist, "test", Integer.MAX_VALUE);
            } else if (mnist != null && classes == 36 && new File(mnist, "mnist").isDirectory() && EmnistLoader.exists(mnist + "/emnist")) {
                all = Alnum.load(mnist + "/mnist", mnist + "/emnist", "train", 60000);   // data folder = the project folder holding mnist/ and emnist/
                tst = Alnum.load(mnist + "/mnist", mnist + "/emnist", "test", Integer.MAX_VALUE);
            } else if (mnist != null && classes == 10 && new File(find(mnist, "train-images-idx3-ubyte")).exists()) {
                all = MnistLoader.loadSource(find(mnist, "train-images-idx3-ubyte"), find(mnist, "train-labels-idx1-ubyte"), 60000);
                tst = MnistLoader.loadSource(find(mnist, "t10k-images-idx3-ubyte"), find(mnist, "t10k-labels-idx1-ubyte"), Integer.MAX_VALUE);
            }
        } else if (mnist != null && new File(find(mnist, "train-images-idx3-ubyte")).exists()) {
            all = MnistLoader.loadSource(find(mnist, "train-images-idx3-ubyte"), find(mnist, "train-labels-idx1-ubyte"), 60000);
            tst = MnistLoader.loadSource(find(mnist, "t10k-images-idx3-ubyte"), find(mnist, "t10k-labels-idx1-ubyte"), Integer.MAX_VALUE);
            System.out.println("no " + modelPath + " yet: training a CNN for " + System.getProperty("minidl.epochs", "3") + " epochs first...");
            model = train(all);
            ModelIO.saveBinary(model, modelPath);
        } else {
            System.out.println("model file '" + modelPath + "' not found.\n  Digits:  java -cp out minidl.MnistDemo mnist 8 cnn aug   (makes mnist-cnn-aug.bin)\n  Letters: ./get-emnist.sh  then  java -cp out minidl.EmnistDemo emnist   (makes emnist-letters-cnn.bin)\n"
                + "  or let the server train a quick one:  java -cp out minidl.HandwritingServer " + modelPath + " 8080 mnist");
            return;
        }
        rec = new DigitRecognizer(model);
        if (all != null) rec.replayFrom(all.view(0, all.size() - all.size() / 10));
        if (tst != null) { testAcc = rec.accuracy(tst); System.out.printf("model %s: %.2f%% on %d unseen test pictures%n", modelPath, testAcc, tst.size()); }
        if (!new File(webDir, "index.html").exists() && HandwritingServer.class.getResource("/web/index.html") == null) System.out.println("warning: " + new File(webDir, "index.html").getAbsolutePath() + " not found (run from the project folder)");

        HttpServer http = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0); // localhost only
        http.createContext("/", ex -> guard(ex, HandwritingServer::page));
        http.createContext("/api/info", ex -> guard(ex, HandwritingServer::info));
        http.createContext("/api/predict", ex -> guard(ex, HandwritingServer::predict));
        http.createContext("/api/learn", ex -> guard(ex, HandwritingServer::learn));
        http.createContext("/api/save", ex -> guard(ex, HandwritingServer::save));
        http.createContext("/api/reset", ex -> guard(ex, HandwritingServer::reset));
        http.setExecutor(Executors.newFixedThreadPool(4));
        http.start();
        System.out.println("open http://localhost:" + port + "   (Ctrl+C to stop)");
    }

    // ------------------------------------------------------------------ handlers
    interface Handler { void run(HttpExchange ex) throws Exception; }

    static void guard(HttpExchange ex, Handler h) throws IOException {
        try { h.run(ex); }
        catch (IllegalArgumentException e) { send(ex, 400, "application/json", "{\"error\":" + str(e.getMessage()) + "}"); }
        catch (Exception e) { send(ex, 500, "application/json", "{\"error\":" + str(String.valueOf(e)) + "}"); }
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
        send(ex, 200, "application/json", "{\"params\":" + rec.paramCount() + ",\"kb\":" + String.format("%.1f", rec.weightBytes() / 1024.0)
            + ",\"alphabet\":" + str(rec.alphabet) + ",\"learned\":" + rec.learnedCount() + ",\"testAcc\":" + testAcc + ",\"model\":" + str(new File(modelPath).getName()) + "}");
    }

    static void predict(HttpExchange ex) throws Exception {
        Map<String, String> q = query(ex);
        int[] wh = new int[2];
        float[] ink = readInk(ex, q, wh);
        boolean multi = "multi".equals(q.get("mode")), tta = !"0".equals(q.get("tta"));
        long t0 = System.nanoTime();
        List<DigitPreprocess.Piece> pieces = new java.util.ArrayList<>();
        if (multi) pieces = DigitPreprocess.segment(ink, wh[0], wh[1], rec.style());
        else { DigitPreprocess.Piece p = DigitPreprocess.whole(ink, wh[0], wh[1], rec.style()); if (p != null) pieces.add(p); }
        StringBuilder digits = new StringBuilder(), text = new StringBuilder();
        for (DigitPreprocess.Piece p : pieces) {
            DigitRecognizer.Result r = rec.predict(p.pixels, tta);
            if (digits.length() > 0) digits.append(',');
            digits.append(digitJson(p, r));
            text.append(r.confidence < 0.5 ? '?' : rec.alphabet.charAt(r.label));
        }
        double ms = (System.nanoTime() - t0) / 1e6;
        send(ex, 200, "application/json", String.format("{\"ms\":%.1f,\"text\":\"%s\",\"digits\":[%s]}", ms, text, digits));
    }

    static void learn(HttpExchange ex) throws Exception {
        Map<String, String> q = query(ex);
        int[] wh = new int[2];
        float[] ink = readInk(ex, q, wh);
        int label = Integer.parseInt(q.getOrDefault("label", "-1"));
        DigitPreprocess.Piece p = DigitPreprocess.whole(ink, wh[0], wh[1], rec.style());
        if (p == null) throw new IllegalArgumentException("nothing drawn");
        double loss = rec.learn(p.pixels, label);
        try (FileOutputStream f = new FileOutputStream(rec.classes == 36 ? "feedback-all.bin" : rec.classes == 26 ? "feedback-letters.bin" : "feedback.bin", true)) { // every correction is kept: 784 pixel bytes + 1 label byte
            byte[] rec1 = new byte[DigitRecognizer.DIM + 1];
            for (int i = 0; i < DigitRecognizer.DIM; i++) rec1[i] = (byte) Math.round(p.pixels[i] * 255);
            rec1[DigitRecognizer.DIM] = (byte) label;
            f.write(rec1);
        }
        send(ex, 200, "application/json", String.format("{\"loss\":%.4f,\"learned\":%d,\"digit\":%s}", loss, rec.learnedCount(), digitJson(p, rec.predict(p.pixels, true))));
    }

    static void save(HttpExchange ex) throws Exception {
        String out = modelPath.replaceAll("\\.bin$", "") + "-tuned.bin";
        rec.save(out);
        send(ex, 200, "application/json", "{\"saved\":" + str(out) + "}");
    }

    static void reset(HttpExchange ex) throws IOException {
        rec.reset();
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

    static String digitJson(DigitPreprocess.Piece p, DigitRecognizer.Result r) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("{\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d,\"label\":%d,\"ch\":\"" + rec.alphabet.charAt(r.label) + "\",\"conf\":%.4f,\"bits\":%.3f,\"probs\":[", p.x, p.y, p.w, p.h, r.label, r.confidence, r.entropyBits));
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

    static String firstExisting(String... names) { for (String n : names) if (new File(n).exists()) return n; return null; }

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
