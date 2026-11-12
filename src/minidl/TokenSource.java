package minidl;

/**
 * Training windows over a sequence of token ids (for example the output of {@link BpeTokenizer#encode}), the same job
 * {@link CharSource} does for characters. Example i = tokens [i, i+T) as input; the target at every position is the NEXT token
 * (one-hot over the vocabulary), so one window teaches T predictions at once.
 */
public class TokenSource implements DataSource {
    private final int[] ids;
    private final int V, T, offset, count;

    private TokenSource(int[] ids, int V, int T, int offset, int count) { this.ids = ids; this.V = V; this.T = T; this.offset = offset; this.count = count; }

    public static TokenSource of(int[] ids, int vocabSize, int T) {
        if (T < 1) throw new IllegalArgumentException("window must be at least 1");
        if (ids.length < T + 2) throw new IllegalArgumentException("the text has " + ids.length + " tokens, shorter than the window of " + T);
        for (int v : ids) if (v < 0 || v >= vocabSize) throw new IllegalArgumentException("token id " + v + " is outside 0.." + (vocabSize - 1));
        return new TokenSource(ids, vocabSize, T, 0, ids.length - T - 1);
    }

    public TokenSource view(int from, int to) {
        if (from < 0 || to > count || from > to) throw new IllegalArgumentException("bad view range");
        return new TokenSource(ids, V, T, offset + from, to - from);
    }

    public int vocabSize() { return V; }
    public int steps() { return T; }
    @Override public int size() { return count; }

    @Override public Tensor features(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, T);
        for (int r = from; r < to; r++) for (int t = 0; t < T; t++) out.data[(r - from) * T + t] = ids[offset + idx[r] + t];
        return out;
    }

    @Override public Tensor targets(int[] idx, int from, int to) {
        Tensor out = new Tensor((to - from) * T, V);
        for (int r = from; r < to; r++) for (int t = 0; t < T; t++) out.data[((r - from) * T + t) * V + ids[offset + idx[r] + t + 1]] = 1f;
        return out;
    }
}
