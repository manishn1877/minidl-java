package minidl;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Draws training curves as an SVG picture (GitHub shows SVG in a README), from the log lines the demos print:
 * {@code epoch    3  loss 0.14079  val 0.05357}. Pure Java, no libraries.
 *
 *   java -jar minidl.jar plot [logs-folder=docs/logs] [out=docs/mnist-training.svg]
 */
public final class Plot {
    private Plot() {}

    /** The curves of one training run. */
    public static final class Series {
        public final String title;
        public final List<Integer> epoch = new ArrayList<>();
        public final List<Double> train = new ArrayList<>(), val = new ArrayList<>();
        Series(String title) { this.title = title; }
    }

    private static final Pattern LINE = Pattern.compile("^epoch\\s+(\\d+)\\s+loss\\s+([0-9.eE+-]+)\\s+val\\s+([0-9.eE+-]+)");

    public static Series parseLog(String title, String logText) {
        Series s = new Series(title);
        for (String line : logText.split("\\R")) {
            Matcher m = LINE.matcher(line.trim());
            if (m.find()) { s.epoch.add(Integer.parseInt(m.group(1))); s.train.add(Double.parseDouble(m.group(2))); s.val.add(Double.parseDouble(m.group(3))); }
        }
        return s;
    }

    private static String esc(String s) { return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;"); }

    private static String f(double v) { return String.format(Locale.ROOT, "%.1f", v); }

    /** One chart per series, side by side. */
    public static String svg(List<Series> panels, String heading) {
        final int pw = 460, ph = 330, left = 58, right = 16, top = 52, bottom = 46;
        int width = pw * Math.max(1, panels.size()), height = ph + 36;
        StringBuilder b = new StringBuilder();
        b.append("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"").append(width).append("\" height=\"").append(height).append("\" viewBox=\"0 0 ").append(width).append(' ').append(height)
         .append("\" font-family=\"sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
        b.append("<text x=\"").append(width / 2).append("\" y=\"22\" text-anchor=\"middle\" font-size=\"15\" font-weight=\"bold\">").append(esc(heading)).append("</text>\n");
        for (int p = 0; p < panels.size(); p++) {
            Series s = panels.get(p);
            int ox = p * pw, x0 = ox + left, x1 = ox + pw - right, y0 = 36 + top, y1 = 36 + ph - bottom;
            double maxLoss = 0.01;
            for (double v : s.train) maxLoss = Math.max(maxLoss, v);
            for (double v : s.val) maxLoss = Math.max(maxLoss, v);
            maxLoss *= 1.08;
            int n = s.epoch.size(), lastEpoch = n == 0 ? 1 : s.epoch.get(n - 1), firstEpoch = n == 0 ? 1 : s.epoch.get(0);
            double xr = Math.max(1, lastEpoch - firstEpoch);
            b.append("<text x=\"").append((x0 + x1) / 2).append("\" y=\"").append(y0 - 14).append("\" text-anchor=\"middle\" font-weight=\"bold\">").append(esc(s.title)).append("</text>\n");
            for (int g = 0; g <= 4; g++) { // grid and y labels
                double v = maxLoss * g / 4, y = y1 - (y1 - y0) * g / 4.0;
                b.append("<line x1=\"").append(x0).append("\" y1=\"").append(f(y)).append("\" x2=\"").append(x1).append("\" y2=\"").append(f(y)).append("\" stroke=\"#e0e0e0\"/>\n");
                b.append("<text x=\"").append(x0 - 6).append("\" y=\"").append(f(y + 4)).append("\" text-anchor=\"end\">").append(String.format(Locale.ROOT, "%.2f", v)).append("</text>\n");
            }
            b.append("<line x1=\"").append(x0).append("\" y1=\"").append(y1).append("\" x2=\"").append(x1).append("\" y2=\"").append(y1).append("\" stroke=\"#444\"/>\n");
            b.append("<line x1=\"").append(x0).append("\" y1=\"").append(y0).append("\" x2=\"").append(x0).append("\" y2=\"").append(y1).append("\" stroke=\"#444\"/>\n");
            for (int i = 0; i < n; i++) {
                double x = n == 1 ? (x0 + x1) / 2.0 : x0 + (x1 - x0) * (s.epoch.get(i) - firstEpoch) / xr;
                b.append("<text x=\"").append(f(x)).append("\" y=\"").append(y1 + 16).append("\" text-anchor=\"middle\">").append(s.epoch.get(i)).append("</text>\n");
            }
            b.append("<text x=\"").append((x0 + x1) / 2).append("\" y=\"").append(y1 + 34).append("\" text-anchor=\"middle\">epoch</text>\n");
            b.append("<text transform=\"translate(").append(ox + 14).append(' ').append((y0 + y1) / 2).append(") rotate(-90)\" text-anchor=\"middle\">loss</text>\n");
            curve(b, s.epoch, s.train, "#1f77b4", x0, x1, y0, y1, firstEpoch, xr, maxLoss);
            curve(b, s.epoch, s.val, "#ff7f0e", x0, x1, y0, y1, firstEpoch, xr, maxLoss);
            b.append("<rect x=\"").append(x1 - 130).append("\" y=\"").append(y0 + 6).append("\" width=\"124\" height=\"38\" fill=\"white\" stroke=\"#ccc\"/>\n");
            b.append("<line x1=\"").append(x1 - 122).append("\" y1=\"").append(y0 + 17).append("\" x2=\"").append(x1 - 100).append("\" y2=\"").append(y0 + 17).append("\" stroke=\"#1f77b4\" stroke-width=\"2\"/><text x=\"").append(x1 - 94).append("\" y=\"").append(y0 + 21).append("\">training loss</text>\n");
            b.append("<line x1=\"").append(x1 - 122).append("\" y1=\"").append(y0 + 34).append("\" x2=\"").append(x1 - 100).append("\" y2=\"").append(y0 + 34).append("\" stroke=\"#ff7f0e\" stroke-width=\"2\"/><text x=\"").append(x1 - 94).append("\" y=\"").append(y0 + 38).append("\">validation loss</text>\n");
        }
        return b.append("</svg>\n").toString();
    }

    private static void curve(StringBuilder b, List<Integer> ep, List<Double> v, String color, int x0, int x1, int y0, int y1, int first, double xr, double maxLoss) {
        if (v.isEmpty()) return;
        StringBuilder pts = new StringBuilder(), dots = new StringBuilder();
        for (int i = 0; i < v.size(); i++) {
            double x = v.size() == 1 ? (x0 + x1) / 2.0 : x0 + (x1 - x0) * (ep.get(i) - first) / xr, y = y1 - (y1 - y0) * v.get(i) / maxLoss;
            pts.append(i > 0 ? " " : "").append(f(x)).append(',').append(f(y));
            dots.append("<circle cx=\"").append(f(x)).append("\" cy=\"").append(f(y)).append("\" r=\"3.5\" fill=\"").append(color).append("\"/>\n");
        }
        b.append("<polyline fill=\"none\" stroke=\"").append(color).append("\" stroke-width=\"2\" points=\"").append(pts).append("\"/>\n").append(dots);
    }

    static String titleOf(String fileName) {
        String t = fileName.replaceAll("\\.log$", "").replace('-', ' ');
        return t.substring(0, 1).toUpperCase(Locale.ROOT) + t.substring(1);
    }

    public static void main(String[] args) throws IOException {
        String dir = args.length > 0 ? args[0] : "docs/logs", out = args.length > 1 ? args[1] : "docs/mnist-training.svg";
        File[] logs = new File(dir).listFiles((d, n) -> n.endsWith(".log"));
        if (logs == null || logs.length == 0) { System.out.println("no .log files in " + dir); return; }
        java.util.Arrays.sort(logs);
        List<Series> panels = new ArrayList<>();
        for (File l : logs) {
            Series s = parseLog(titleOf(l.getName()), new String(Files.readAllBytes(l.toPath()), StandardCharsets.UTF_8));
            if (!s.epoch.isEmpty()) panels.add(s);
        }
        File parent = new File(out).getAbsoluteFile().getParentFile();
        if (parent != null) parent.mkdirs();
        Files.write(Paths.get(out), svg(panels, "MiniDL: loss per epoch (from " + dir + ")").getBytes(StandardCharsets.UTF_8));
        System.out.println("wrote " + out + " (" + panels.size() + " charts)");
    }
}
