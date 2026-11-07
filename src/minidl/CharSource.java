package minidl;

import java.util.TreeSet;

/**
 * Text for a character-level language model. Every window of T characters is one training example:
 * the model sees characters s..s+T-1 and must predict, at EVERY position, the character that follows.
 * features = (rows x T) character ids.   targets = (rows*T x vocab) one-hot of the next character.
 * Memory: one byte per character.
 */
public class CharSource implements DataSource {
    private final byte[] ids;
    private final char[] vocab;
    private final int T, offset, count;

    private CharSource(byte[] ids, char[] vocab, int T, int offset, int count) {
        this.ids = ids; this.vocab = vocab; this.T = T; this.offset = offset; this.count = count;
    }

    public static CharSource fromText(String text, int T) {
        TreeSet<Character> set = new TreeSet<>();
        for (char c : text.toCharArray()) set.add(c);
        if (set.size() > 255) throw new IllegalArgumentException("too many different characters (" + set.size() + ")");
        char[] vocab = new char[set.size()];
        int k = 0;
        for (char c : set) vocab[k++] = c;
        byte[] ids = new byte[text.length()];
        for (int i = 0; i < ids.length; i++) ids[i] = (byte) java.util.Arrays.binarySearch(vocab, text.charAt(i));
        if (ids.length < T + 2) throw new IllegalArgumentException("text is shorter than the window");
        return new CharSource(ids, vocab, T, 0, ids.length - T - 1);
    }

    /** Windows [from, to). */
    public CharSource view(int from, int to) {
        if (from < 0 || to > count || from > to) throw new IllegalArgumentException("bad view range");
        return new CharSource(ids, vocab, T, offset + from, to - from);
    }

    public int vocabSize() { return vocab.length; }
    public char[] vocab() { return vocab.clone(); }
    public int steps() { return T; }
    public int id(char c) { int i = java.util.Arrays.binarySearch(vocab, c); if (i < 0) throw new IllegalArgumentException("unknown character '" + c + "'"); return i; }
    public char charOf(int id) { return vocab[id]; }

    @Override public int size() { return count; }

    @Override public Tensor features(int[] idx, int from, int to) {
        Tensor out = new Tensor(to - from, T);
        for (int r = from; r < to; r++) {
            int s = offset + idx[r];
            for (int t = 0; t < T; t++) out.data[(r - from) * T + t] = ids[s + t] & 0xFF;
        }
        return out;
    }

    @Override public Tensor targets(int[] idx, int from, int to) {
        int V = vocab.length;
        Tensor out = new Tensor((to - from) * T, V);
        for (int r = from; r < to; r++) {
            int s = offset + idx[r];
            for (int t = 0; t < T; t++) out.data[((r - from) * T + t) * V + (ids[s + t + 1] & 0xFF)] = 1f;
        }
        return out;
    }
}
