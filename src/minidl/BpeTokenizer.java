package minidl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-pair-encoding (BPE) tokenizer: a learned vocabulary of sub-words, the kind GPT-style models use.
 *
 * Training: start with every distinct character as a token. Count how often two tokens sit side by side (inside one word,
 * a word keeps its leading space), merge the most frequent pair into a NEW token, repeat until the vocabulary has the size you
 * asked for. Common words become one token ("the"), rare words become a few pieces ("zeb" "ra").
 * Encoding applies the learned merges in the order they were learned. Decoding just glues the token texts together,
 * so decode(encode(text)) == text for every text whose characters were seen in training (an unseen character becomes
 * token 0, the "unknown" token, and decodes to U+FFFD).
 * A word-piece tokenizer is the same idea with a different merge score, so this is the tokenizer step for the transformer.
 */
public final class BpeTokenizer {
    public static final int UNKNOWN = 0;
    private static final Pattern PRE = Pattern.compile(" ?[^\\s]+|\\s+"); // a word with its leading space, or a run of spaces/newlines

    private final List<String> vocab = new ArrayList<>();             // id -> text, id 0 = unknown
    private final Map<String, Integer> id = new HashMap<>();
    private final List<int[]> merges = new ArrayList<>();             // (left id, right id), in the order learned; result = vocab id (base + index)
    private final Map<Long, Integer> rank = new HashMap<>();          // pair -> merge index
    private final Map<Long, Integer> mergedId = new HashMap<>();      // pair -> id of the merged token

    private BpeTokenizer() { vocab.add("\uFFFD"); }

    private static long key(int a, int b) { return ((long) a << 32) | (b & 0xffffffffL); }

    public int vocabSize() { return vocab.size(); }
    public String token(int i) { return vocab.get(i); }

    /** Learns a vocabulary of at most vocabSize tokens (characters + merges) from the text. */
    public static BpeTokenizer train(String text, int vocabSize) {
        BpeTokenizer t = new BpeTokenizer();
        Map<String, Integer> words = new LinkedHashMap<>();
        Matcher m = PRE.matcher(text);
        while (m.find()) words.merge(m.group(), 1, Integer::sum);
        for (int i = 0; i < text.length(); ) { // base vocabulary: every character, in order of first appearance
            int cp = text.codePointAt(i);
            String s = new String(Character.toChars(cp));
            if (!t.id.containsKey(s)) { t.id.put(s, t.vocab.size()); t.vocab.add(s); }
            i += Character.charCount(cp);
        }
        List<int[]> seqs = new ArrayList<>();
        List<Integer> freq = new ArrayList<>();
        for (Map.Entry<String, Integer> e : words.entrySet()) {
            String w = e.getKey();
            int[] ids = new int[w.codePointCount(0, w.length())];
            for (int i = 0, k = 0; i < w.length(); k++) { int cp = w.codePointAt(i); ids[k] = t.id.get(new String(Character.toChars(cp))); i += Character.charCount(cp); }
            seqs.add(ids); freq.add(e.getValue());
        }
        while (t.vocab.size() < vocabSize) {
            Map<Long, Long> counts = new HashMap<>();
            for (int w = 0; w < seqs.size(); w++) { int[] s = seqs.get(w); for (int i = 0; i + 1 < s.length; i++) counts.merge(key(s[i], s[i + 1]), (long) freq.get(w), Long::sum); }
            long bestKey = -1, best = 1; // a pair seen once is not worth a token
            for (Map.Entry<Long, Long> e : counts.entrySet())
                if (e.getValue() > best || (e.getValue() == best && bestKey >= 0 && e.getKey() < bestKey)) { best = e.getValue(); bestKey = e.getKey(); }
            if (bestKey < 0) break; // nothing left to merge
            int a = (int) (bestKey >> 32), b = (int) bestKey, nid = t.vocab.size();
            t.vocab.add(t.vocab.get(a) + t.vocab.get(b));
            t.id.put(t.vocab.get(nid), nid);
            t.rank.put(bestKey, t.merges.size()); t.mergedId.put(bestKey, nid); t.merges.add(new int[]{a, b});
            for (int w = 0; w < seqs.size(); w++) seqs.set(w, mergeIn(seqs.get(w), a, b, nid));
        }
        return t;
    }

    private static int[] mergeIn(int[] s, int a, int b, int nid) {
        int[] out = new int[s.length];
        int n = 0;
        for (int i = 0; i < s.length; i++) {
            if (i + 1 < s.length && s[i] == a && s[i + 1] == b) { out[n++] = nid; i++; } else out[n++] = s[i];
        }
        return java.util.Arrays.copyOf(out, n);
    }

    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        Matcher m = PRE.matcher(text);
        while (m.find()) {
            String w = m.group();
            List<Integer> s = new ArrayList<>();
            for (int i = 0; i < w.length(); ) { int cp = w.codePointAt(i); Integer v = id.get(new String(Character.toChars(cp))); s.add(v == null ? UNKNOWN : v); i += Character.charCount(cp); }
            while (s.size() > 1) { // apply the earliest-learned merge that is present
                int bestRank = Integer.MAX_VALUE, at = -1;
                for (int i = 0; i + 1 < s.size(); i++) { Integer r = rank.get(key(s.get(i), s.get(i + 1))); if (r != null && r < bestRank) { bestRank = r; at = i; } }
                if (at < 0) break;
                s.set(at, mergedId.get(key(s.get(at), s.get(at + 1))));
                s.remove(at + 1);
            }
            out.addAll(s);
        }
        int[] r = new int[out.size()];
        for (int i = 0; i < r.length; i++) r[i] = out.get(i);
        return r;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int i : ids) {
            if (i < 0 || i >= vocab.size()) throw new IllegalArgumentException("token id " + i + " is outside 0.." + (vocab.size() - 1));
            sb.append(vocab.get(i));
        }
        return sb.toString();
    }

    /** Text file: first line MDLBPE1, then the vocabulary (one token per line as code points), then the merges ("left right"). */
    public void save(String path) throws IOException {
        StringBuilder sb = new StringBuilder("MDLBPE1\n").append(vocab.size()).append('\n');
        for (String v : vocab) { StringBuilder line = new StringBuilder(); v.codePoints().forEach(cp -> line.append(line.length() > 0 ? "," : "").append(cp)); sb.append(line).append('\n'); }
        sb.append(merges.size()).append('\n');
        for (int[] mg : merges) sb.append(mg[0]).append(' ').append(mg[1]).append('\n');
        Files.write(Paths.get(path), sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    public static BpeTokenizer load(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
        try {
            if (lines.isEmpty() || !lines.get(0).equals("MDLBPE1")) throw new IOException("not a MiniDL tokenizer file");
            BpeTokenizer t = new BpeTokenizer();
            t.vocab.clear();
            int nv = Integer.parseInt(lines.get(1)), p = 2;
            if (nv < 1 || nv > 10_000_000) throw new IOException("bad vocabulary size " + nv);
            for (int i = 0; i < nv; i++, p++) {
                StringBuilder sb = new StringBuilder();
                for (String cp : lines.get(p).split(",")) sb.appendCodePoint(Integer.parseInt(cp));
                t.vocab.add(sb.toString());
                if (i > 0) t.id.put(sb.toString(), i);
            }
            int nm = Integer.parseInt(lines.get(p++));
            for (int i = 0; i < nm; i++, p++) {
                String[] ab = lines.get(p).split(" ");
                int a = Integer.parseInt(ab[0]), b = Integer.parseInt(ab[1]);
                if (a < 0 || b < 0 || a >= nv || b >= nv) throw new IOException("merge refers to a missing token");
                int nid = t.id.get(t.vocab.get(a) + t.vocab.get(b));
                t.rank.put(key(a, b), i); t.mergedId.put(key(a, b), nid); t.merges.add(new int[]{a, b});
            }
            return t;
        } catch (RuntimeException e) { throw new IOException("damaged tokenizer file: " + e, e); }
    }
}
