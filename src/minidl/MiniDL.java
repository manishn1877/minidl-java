package minidl;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Entry point of minidl.jar:  {@code java -jar minidl.jar <command> [arguments]}.
 * Also holds the library version.
 */
public final class MiniDL {
    private MiniDL() {}

    /** Library version (kept in step with pom.xml). */
    public static final String VERSION = "1.3.0";

    private static final Map<String, String[]> COMMANDS = new LinkedHashMap<>();
    static {
        COMMANDS.put("demo",        new String[]{"Main", "gradient check, XOR, spiral, delivery-time regression"});
        COMMANDS.put("tests",       new String[]{"Tests", "the full automated test suite"});
        COMMANDS.put("mnist",       new String[]{"MnistDemo", "train handwritten digits: <mnist-folder> [epochs] [mlp|cnn] [limit] [aug] [mmap]"});
        COMMANDS.put("emnist",      new String[]{"EmnistDemo", "train handwritten letters: <emnist-folder> [epochs] [limit] [aug]"});
        COMMANDS.put("alnum",       new String[]{"AlnumDemo", "ONE model for digits and letters: <mnist-folder> <emnist-folder> [epochs] [mnistLimit] [aug]"});
        COMMANDS.put("check-data",  new String[]{"DataCheck", "SEE the training pictures per class, to check labels: <emnist-folder> [mnist-folder]"});
        COMMANDS.put("plot",        new String[]{"Plot", "training curves as SVG: [logs-folder=docs/logs] [out=docs/mnist-training.svg]"});
        COMMANDS.put("site",        new String[]{"SiteBuilder", "build the GitHub Pages site into docs/"});
        COMMANDS.put("cifar",       new String[]{"CifarDemo", "train on CIFAR-10 colour photos"});
        COMMANDS.put("text",        new String[]{"CharTransformerDemo", "tiny GPT-style character model"});
        COMMANDS.put("precision",   new String[]{"MixedPrecisionDemo", "16-bit mixed-precision training"});
        COMMANDS.put("bench",       new String[]{"Benchmark", "speed and memory benchmarks"});
        COMMANDS.put("serve",       new String[]{"ModelServer", "model server (DJL-style): [--port N] [--store DIR] [--model name=file.bin,batch=16,delay=5,workers=1,max_workers=4,queue=100]"});
        COMMANDS.put("web",         new String[]{"HandwritingServer", "handwriting web app: [auto|model.bin] [port] [data-folder]"});
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("help") || args[0].equals("--help") || args[0].equals("-h")) { usage(); return; }
        if (args[0].equals("version") || args[0].equals("--version")) { System.out.println("MiniDL " + VERSION + "  (Java " + System.getProperty("java.version") + ", " + Tensor.kernelName() + ")"); return; }
        String[] cmd = COMMANDS.get(args[0]);
        if (cmd == null) { System.out.println("unknown command '" + args[0] + "'\n"); usage(); System.exit(2); return; }
        try {
            Class.forName("minidl." + cmd[0]).getMethod("main", String[].class).invoke(null, (Object) Arrays.copyOfRange(args, 1, args.length));
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof Exception) throw (Exception) e.getCause();
            throw e;
        }
    }

    private static void usage() {
        System.out.println("MiniDL " + VERSION + ": a tiny deep learning library in pure Java\n\nusage: java -jar minidl.jar <command> [arguments]\n");
        for (Map.Entry<String, String[]> e : COMMANDS.entrySet()) System.out.printf("  %-10s %s%n", e.getKey(), e.getValue()[1]);
        System.out.println("  version    print the version\n\nTo use it in your own code, put minidl.jar on the classpath and import minidl.*  (see README section 9).");
    }
}
