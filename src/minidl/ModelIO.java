package minidl;

import java.io.*;
import java.util.*;

/**
 * Save / load a model as a plain text file.
 * Version 2 stores the architecture too, so you can do:
 *     Sequential m = ModelIO.loadModel("model.txt", new Random());
 * without rebuilding the layers by hand.
 * (Custom layers you wrote yourself can still be saved, but loadModel needs a built-in layer type.
 *  For those, build the model yourself and use load(model, path).)
 */
public final class ModelIO {
    private ModelIO() {}

    public static void save(Sequential model, String path) throws IOException {
        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {
            w.println("MINIDL 2");
            List<Layer> ls = model.layers();
            w.println("layers " + ls.size());
            for (Layer l : ls) w.println(l.spec());
            List<Param> ps = model.params();
            w.println("params " + ps.size());
            for (Param p : ps) writeTensor(w, p.value);
            List<Tensor> st = model.state();
            w.println("state " + st.size());
            for (Tensor t : st) writeTensor(w, t);
        }
    }

    /** Build the model from the file, then fill in the weights. */
    public static Sequential loadModel(String path, Random rnd) throws IOException {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            expect(r.readLine(), "MINIDL 2");
            int n = countLine(r.readLine(), "layers");
            Sequential m = new Sequential();
            for (int i = 0; i < n; i++) m.add(build(r.readLine(), rnd));
            readWeights(m, r);
            return m;
        }
    }

    /** Load weights into a model you already built. The architecture must match the file. */
    public static void load(Sequential model, String path) throws IOException {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String first = r.readLine();
            if (first == null) throw new IOException("empty file");
            if (first.trim().equals("MINIDL 2")) {
                int n = countLine(r.readLine(), "layers");
                List<Layer> ls = model.layers();
                if (n != ls.size()) throw new IOException("architecture mismatch: file has " + n + " layers, model has " + ls.size());
                for (int i = 0; i < n; i++) {
                    String spec = r.readLine().trim();
                    if (!spec.equals(ls.get(i).spec()))
                        throw new IOException("architecture mismatch at layer " + (i + 1) + ": file has '" + spec + "', model has '" + ls.get(i).spec() + "'");
                }
                readWeights(model, r);
            } else { // old version 1 file: just weights
                List<Param> ps = model.params();
                if (Integer.parseInt(first.trim()) != ps.size()) throw new IOException("layer count mismatch");
                for (Param p : ps) readTensorInto(r, p.value);
            }
        }
    }

    private static void readWeights(Sequential m, BufferedReader r) throws IOException {
        List<Param> ps = m.params();
        if (countLine(r.readLine(), "params") != ps.size()) throw new IOException("parameter count mismatch");
        for (Param p : ps) readTensorInto(r, p.value);
        List<Tensor> st = m.state();
        if (countLine(r.readLine(), "state") != st.size()) throw new IOException("state count mismatch");
        for (Tensor t : st) readTensorInto(r, t);
    }

    /** Builds a layer from the words of its saved description ("AutoDense 64 32" -> {"AutoDense", "64", "32"}). */
    public interface LayerFactory { Layer create(String[] words, Random rnd); }

    private static final java.util.Map<String, LayerFactory> REGISTRY = new java.util.concurrent.ConcurrentHashMap<>();

    /** Teach loadModel about your own layer type. The name is the first word of the layer's spec(). */
    public static void register(String name, LayerFactory factory) { REGISTRY.put(name, factory); }

    private static Layer build(String spec, Random rnd) throws IOException {
        spec = spec.trim();
        int br = spec.indexOf('[');
        if (br >= 0) { // blocks that contain other layers:  Name args [ layer | layer | layer ]
            String[] t = spec.substring(0, br).trim().split("\\s+");
            String body = spec.substring(br + 1, spec.lastIndexOf(']'));
            List<Layer> inner = new ArrayList<>();
            for (String part : splitTop(body)) inner.add(build(part, rnd));
            switch (t[0]) {
                case "Residual":  return new Residual(inner);
                case "TokenWise": return new TokenWise(i(t[1]), i(t[2]), inner);
                default: throw new IOException("unknown block type '" + t[0] + "'");
            }
        }
        String[] t = spec.split("\\s+");
        switch (t[0]) {
            case "Dense":      return new Dense(i(t[1]), i(t[2]), rnd);
            case "ReLU":       return new Activations.ReLU();
            case "LeakyReLU":  return new Activations.LeakyReLU(Double.parseDouble(t[1]));
            case "Sigmoid":    return new Activations.Sigmoid();
            case "Tanh":       return new Activations.Tanh();
            case "GELU":       return new Activations.GELU();
            case "Dropout":    return new Dropout(Double.parseDouble(t[1]), rnd);
            case "BatchNorm":  return new BatchNorm(i(t[1]), Double.parseDouble(t[2]));
            case "LayerNorm":  return new LayerNorm(i(t[1]));
            case "BatchNorm2D": return new BatchNorm2D(i(t[1]), i(t[2]), i(t[3]), Double.parseDouble(t[4]));
            case "LSTM":       return new LSTM(i(t[1]), i(t[2]), i(t[3]), rnd);
            case "Conv2D":     return t.length > 8 ? new Conv2D(i(t[1]), i(t[2]), i(t[3]), i(t[4]), i(t[5]), i(t[6]), i(t[7]), i(t[8]), i(t[9]), rnd)
                                           : new Conv2D(i(t[1]), i(t[2]), i(t[3]), i(t[4]), i(t[5]), i(t[6]), i(t[7]), rnd);
            case "MaxPool2D":  return new MaxPool2D(i(t[1]), i(t[2]), i(t[3]), i(t[4]));
            case "AvgPool2D":  return new AvgPool2D(i(t[1]), i(t[2]), i(t[3]), i(t[4]));
            case "Embedding":  return new Embedding(i(t[1]), i(t[2]), i(t[3]), rnd);
            case "CausalSelfAttention": return new CausalSelfAttention(i(t[1]), i(t[2]), i(t[3]), rnd);
            case "Reshape":    return new Reshape(i(t[1]));
            case "QDense":     return new QuantizedDense(i(t[1]), i(t[2]));
            case "QDense4":    return new QuantizedDense4(i(t[1]), i(t[2]), i(t[3]));
            case "BF16Dense":  return new BF16Dense(i(t[1]), i(t[2]));
            case "QConv2D":    return new QuantizedConv2D(i(t[1]), i(t[2]), i(t[3]), i(t[4]), i(t[5]), i(t[6]), i(t[7]));
            default:
                LayerFactory f = REGISTRY.get(t[0]);
                if (f == null) { // layer families that register themselves when their class loads
                    try { Class.forName("minidl.AutoLayers"); } catch (ClassNotFoundException ignored) { }
                    f = REGISTRY.get(t[0]);
                }
                if (f != null) return f.create(t, rnd);
                throw new IOException("unknown layer type '" + t[0] + "'. Build the model yourself and use ModelIO.load(model, path), or ModelIO.register(...) a factory.");
        }
    }

    /** Splits "a | b [c | d] | e" at the | signs that are not inside [ ]. */
    private static List<String> splitTop(String body) {
        List<String> parts = new ArrayList<>();
        int depth = 0, start = 0;
        for (int k = 0; k < body.length(); k++) {
            char c = body.charAt(k);
            if (c == '[') depth++;
            else if (c == ']') depth--;
            else if (c == '|' && depth == 0) { parts.add(body.substring(start, k)); start = k + 1; }
        }
        parts.add(body.substring(start));
        return parts;
    }

    private static int i(String s) { return Integer.parseInt(s); }

    // ------------------------------------------------------------------ binary format (much smaller, much faster)

    private static final int MAGIC = 0x4D444C33; // "MDL3"

    /** Compact binary file: 4 bytes per weight instead of ~12 characters of text. Also stores int8 layers. */
    public static void saveBinary(Sequential model, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path), 1 << 16))) {
            out.writeInt(MAGIC);
            List<Layer> ls = model.layers();
            out.writeInt(ls.size());
            for (Layer l : ls) out.writeUTF(l.spec());
            List<Param> ps = model.params();
            out.writeInt(ps.size());
            for (Param p : ps) writeBin(out, p.value);
            List<Tensor> st = model.state();
            out.writeInt(st.size());
            for (Tensor t : st) writeBin(out, t);
            for (Layer l : ls) l.writeExtra(out);
        }
    }

    public static Sequential loadModelBinary(String path, Random rnd) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path), 1 << 16))) {
            if (in.readInt() != MAGIC) throw new IOException("not a MiniDL binary model file");
            int n = in.readInt();
            Sequential m = new Sequential();
            for (int k = 0; k < n; k++) m.add(build(in.readUTF(), rnd));
            List<Param> ps = m.params();
            if (in.readInt() != ps.size()) throw new IOException("parameter count mismatch");
            for (Param p : ps) readBin(in, p.value);
            List<Tensor> st = m.state();
            if (in.readInt() != st.size()) throw new IOException("state count mismatch");
            for (Tensor t : st) readBin(in, t);
            for (Layer l : m.layers()) l.readExtra(in);
            return m;
        }
    }

    private static void writeBin(DataOutputStream out, Tensor t) throws IOException {
        out.writeInt(t.rows); out.writeInt(t.cols);
        for (float v : t.data) out.writeFloat(v);
    }

    private static void readBin(DataInputStream in, Tensor t) throws IOException {
        if (in.readInt() != t.rows || in.readInt() != t.cols) throw new IOException("layer size mismatch");
        for (int k = 0; k < t.data.length; k++) t.data[k] = in.readFloat();
    }

    private static void writeTensor(PrintWriter w, Tensor t) {
        w.println(t.rows + " " + t.cols);
        StringBuilder sb = new StringBuilder(t.data.length * 12);
        for (float v : t.data) sb.append(v).append(' ');
        w.println(sb.toString().trim());
    }

    private static void readTensorInto(BufferedReader r, Tensor t) throws IOException {
        String[] shape = r.readLine().trim().split(" ");
        if (Integer.parseInt(shape[0]) != t.rows || Integer.parseInt(shape[1]) != t.cols)
            throw new IOException("layer size mismatch");
        String[] vals = r.readLine().trim().split(" ");
        if (vals.length != t.data.length) throw new IOException("corrupt file: wrong number of values");
        for (int i = 0; i < vals.length; i++) t.data[i] = Float.parseFloat(vals[i]);
    }

    private static void expect(String line, String want) throws IOException {
        if (line == null || !line.trim().equals(want)) throw new IOException("not a MiniDL v2 model file");
    }

    private static int countLine(String line, String key) throws IOException {
        if (line == null) throw new IOException("unexpected end of file");
        String[] t = line.trim().split(" ");
        if (t.length != 2 || !t[0].equals(key)) throw new IOException("expected '" + key + " N' but found '" + line + "'");
        return Integer.parseInt(t[1]);
    }
}
