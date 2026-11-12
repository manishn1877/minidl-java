package minidl;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads and writes the <b>safetensors</b> file format, in plain Java. It is the simple, safe format the PyTorch world
 * uses to share weights (a file holds only numbers, never code), so a model trained in PyTorch can be loaded into MiniDL
 * and the other way round, WITHOUT any Python in this project.
 *
 * File layout: 8 bytes little-endian = length N of the header; N bytes of JSON header
 * {"name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]}, ...}; then the raw numbers (little-endian).
 * This class reads F32 tensors (and keeps I64 ones as raw bytes) and writes F32 and I64.
 *
 * {@link #exportModel} / {@link #importModel} map a MiniDL Sequential to PyTorch names and layouts:
 * <pre>
 *   Dense      "{i}.weight" (out, in)  (MiniDL stores in x out, so it is transposed)   "{i}.bias" (out)
 *   Conv2D     "{i}.weight" (out, in/groups, k, k)                                       "{i}.bias" (out)
 *   BatchNorm2D "{i}.weight" "{i}.bias" "{i}.running_mean" "{i}.running_var" (channels), "{i}.num_batches_tracked"
 * </pre>
 * {i} is the position of the layer in the MiniDL Sequential, counting every layer (also ReLU and pooling), which is exactly the
 * index of an {@code nn.Sequential} built layer by layer in the same order (leave out nn.Flatten: MiniDL needs none).
 * Not tested against a file written by real PyTorch (none was available when this was written): the layouts follow the PyTorch documentation.
 */
public final class SafeTensors {
    private SafeTensors() {}

    /** One tensor of a file. */
    public static final class Entry {
        public final String dtype;
        public final long[] shape;
        public final byte[] raw; // little-endian bytes

        public Entry(String dtype, long[] shape, byte[] raw) { this.dtype = dtype; this.shape = shape; this.raw = raw; }

        public static Entry ofFloats(long[] shape, float[] v) {
            ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : v) b.putFloat(f);
            return new Entry("F32", shape, b.array());
        }

        public static Entry ofLong(long value) {
            return new Entry("I64", new long[0], ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array());
        }

        public float[] floats() {
            if (!dtype.equals("F32")) throw new IllegalArgumentException("tensor is " + dtype + ", not F32");
            float[] v = new float[raw.length / 4];
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(v);
            return v;
        }

        public long count() { long n = 1; for (long d : shape) n *= d; return n; }
    }

    public static void write(String path, Map<String, Entry> tensors) throws IOException {
        StringBuilder h = new StringBuilder("{");
        long offset = 0;
        boolean first = true;
        for (Map.Entry<String, Entry> e : tensors.entrySet()) {
            Entry t = e.getValue();
            if (!first) h.append(',');
            first = false;
            h.append(MiniJson.quote(e.getKey())).append(":{\"dtype\":\"").append(t.dtype).append("\",\"shape\":[");
            for (int i = 0; i < t.shape.length; i++) h.append(i > 0 ? "," : "").append(t.shape[i]);
            h.append("],\"data_offsets\":[").append(offset).append(',').append(offset + t.raw.length).append("]}");
            offset += t.raw.length;
        }
        h.append('}');
        while (h.length() % 8 != 0) h.append(' '); // the format asks for an 8-byte aligned header
        byte[] header = h.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(8 + header.length + (int) offset).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(header.length).put(header);
        for (Entry t : tensors.values()) out.put(t.raw);
        Files.write(Paths.get(path), out.array());
    }

    public static Map<String, Entry> read(String path) throws IOException {
        byte[] all = Files.readAllBytes(Paths.get(path));
        if (all.length < 8) throw new IOException("not a safetensors file: too short");
        long n = ByteBuffer.wrap(all, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (n < 2 || n > all.length - 8L || n > 100_000_000L) throw new IOException("damaged safetensors header length " + n);
        Object parsed;
        try { parsed = MiniJson.parse(new String(all, 8, (int) n, StandardCharsets.UTF_8)); }
        catch (IllegalArgumentException e) { throw new IOException("damaged safetensors header: " + e.getMessage(), e); }
        if (!(parsed instanceof Map)) throw new IOException("safetensors header must be a JSON object");
        long dataStart = 8 + n, dataLen = all.length - dataStart;
        Map<String, Entry> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            String name = (String) e.getKey();
            if (name.equals("__metadata__")) continue;
            if (!(e.getValue() instanceof Map)) throw new IOException("tensor " + name + " is not an object");
            Map<?, ?> t = (Map<?, ?>) e.getValue();
            try {
                String dtype = (String) t.get("dtype");
                List<?> sh = (List<?>) t.get("shape"), off = (List<?>) t.get("data_offsets");
                long[] shape = new long[sh.size()];
                for (int i = 0; i < shape.length; i++) shape[i] = ((Number) sh.get(i)).longValue();
                long b = ((Number) off.get(0)).longValue(), en = ((Number) off.get(1)).longValue();
                if (b < 0 || en < b || en > dataLen) throw new IOException("tensor " + name + " points outside the file");
                byte[] raw = new byte[(int) (en - b)];
                System.arraycopy(all, (int) (dataStart + b), raw, 0, raw.length);
                Entry entry = new Entry(dtype, shape, raw);
                int size = dtype.equals("F32") || dtype.equals("I32") ? 4 : dtype.equals("I64") || dtype.equals("F64") ? 8 : dtype.equals("F16") || dtype.equals("BF16") || dtype.equals("I16") ? 2 : 1;
                if (entry.count() * size != raw.length) throw new IOException("tensor " + name + ": shape " + java.util.Arrays.toString(shape) + " does not fit " + raw.length + " bytes of " + dtype);
                out.put(name, entry);
            } catch (ClassCastException | NullPointerException | IndexOutOfBoundsException ex) { throw new IOException("damaged description of tensor " + name, ex); }
        }
        return out;
    }

    // ---------------------------------------------------------------- Sequential <-> PyTorch layout

    public static void exportModel(Sequential m, String path) throws IOException {
        Map<String, Entry> out = new LinkedHashMap<>();
        List<Layer> layers = m.layers();
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            List<Param> ps = l.params();
            if (ps.isEmpty()) continue;
            String[] spec = l.spec().split(" ");
            switch (spec[0]) {
                case "Dense": {
                    Tensor w = ps.get(0).value, b = ps.get(1).value; // (in, out)
                    float[] t = new float[w.data.length];
                    for (int r = 0; r < w.rows; r++) for (int c = 0; c < w.cols; c++) t[c * w.rows + r] = w.data[r * w.cols + c];
                    out.put(i + ".weight", Entry.ofFloats(new long[]{w.cols, w.rows}, t));
                    out.put(i + ".bias", Entry.ofFloats(new long[]{b.cols}, b.data.clone()));
                    break;
                }
                case "Conv2D": {
                    int k = Integer.parseInt(spec[5]);
                    Tensor w = ps.get(0).value, b = ps.get(1).value; // (out, in/groups*k*k)
                    out.put(i + ".weight", Entry.ofFloats(new long[]{w.rows, w.cols / (k * k), k, k}, w.data.clone()));
                    out.put(i + ".bias", Entry.ofFloats(new long[]{b.cols}, b.data.clone()));
                    break;
                }
                case "BatchNorm2D": {
                    List<Tensor> st = l.state();
                    out.put(i + ".weight", Entry.ofFloats(new long[]{ps.get(0).value.cols}, ps.get(0).value.data.clone()));
                    out.put(i + ".bias", Entry.ofFloats(new long[]{ps.get(1).value.cols}, ps.get(1).value.data.clone()));
                    out.put(i + ".running_mean", Entry.ofFloats(new long[]{st.get(0).cols}, st.get(0).data.clone()));
                    out.put(i + ".running_var", Entry.ofFloats(new long[]{st.get(1).cols}, st.get(1).data.clone()));
                    out.put(i + ".num_batches_tracked", Entry.ofLong(0));
                    break;
                }
                default: throw new IOException("layer " + i + " (" + spec[0] + ") has weights but no PyTorch mapping yet; supported: Dense, Conv2D, BatchNorm2D");
            }
        }
        write(path, out);
    }

    /** Loads the weights of a safetensors file into a model with the SAME layers; every name and shape is checked. */
    public static void importModel(Sequential m, String path) throws IOException {
        Map<String, Entry> in = read(path);
        List<Layer> layers = m.layers();
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            List<Param> ps = l.params();
            if (ps.isEmpty()) continue;
            String[] spec = l.spec().split(" ");
            switch (spec[0]) {
                case "Dense": {
                    Tensor w = ps.get(0).value;
                    float[] src = get(in, i + ".weight", new long[]{w.cols, w.rows});
                    for (int r = 0; r < w.rows; r++) for (int c = 0; c < w.cols; c++) w.data[r * w.cols + c] = src[c * w.rows + r];
                    System.arraycopy(get(in, i + ".bias", new long[]{ps.get(1).value.cols}), 0, ps.get(1).value.data, 0, ps.get(1).value.data.length);
                    break;
                }
                case "Conv2D": {
                    int k = Integer.parseInt(spec[5]);
                    Tensor w = ps.get(0).value;
                    System.arraycopy(get(in, i + ".weight", new long[]{w.rows, w.cols / (k * k), k, k}), 0, w.data, 0, w.data.length);
                    System.arraycopy(get(in, i + ".bias", new long[]{ps.get(1).value.cols}), 0, ps.get(1).value.data, 0, ps.get(1).value.data.length);
                    break;
                }
                case "BatchNorm2D": {
                    List<Tensor> st = l.state();
                    long[] sh = {ps.get(0).value.cols};
                    System.arraycopy(get(in, i + ".weight", sh), 0, ps.get(0).value.data, 0, (int) sh[0]);
                    System.arraycopy(get(in, i + ".bias", sh), 0, ps.get(1).value.data, 0, (int) sh[0]);
                    System.arraycopy(get(in, i + ".running_mean", sh), 0, st.get(0).data, 0, (int) sh[0]);
                    System.arraycopy(get(in, i + ".running_var", sh), 0, st.get(1).data, 0, (int) sh[0]);
                    break;
                }
                default: throw new IOException("layer " + i + " (" + spec[0] + ") has weights but no PyTorch mapping yet");
            }
        }
    }

    private static float[] get(Map<String, Entry> in, String name, long[] shape) throws IOException {
        Entry e = in.get(name);
        if (e == null) throw new IOException("the file has no tensor '" + name + "' (found: " + new ArrayList<>(in.keySet()).subList(0, Math.min(6, in.size())) + " ...)");
        if (!java.util.Arrays.equals(e.shape, shape)) throw new IOException("tensor '" + name + "' has shape " + java.util.Arrays.toString(e.shape) + " but the model needs " + java.util.Arrays.toString(shape));
        return e.floats();
    }
}
