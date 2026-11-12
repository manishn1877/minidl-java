package minidl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A tiny JSON reader (objects, arrays, numbers, strings, true/false/null) for the serving API, so the library stays free of
 * dependencies. Objects become {@code Map<String,Object>}, arrays {@code List<Object>}, numbers {@code Double}.
 * Wrong JSON gives an IllegalArgumentException that says where.
 */
public final class MiniJson {
    private final String s;
    private int i;

    private MiniJson(String s) { this.s = s; }

    public static Object parse(String text) {
        MiniJson p = new MiniJson(text);
        p.ws();
        Object v = p.value(0);
        p.ws();
        if (p.i != text.length()) throw p.fail("extra text after the JSON value");
        return v;
    }

    private IllegalArgumentException fail(String why) { return new IllegalArgumentException("bad JSON at position " + i + ": " + why); }

    private void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

    private Object value(int depth) {
        if (depth > 64) throw fail("nested too deeply");
        if (i >= s.length()) throw fail("unexpected end");
        char c = s.charAt(i);
        if (c == '{') {
            Map<String, Object> m = new LinkedHashMap<>();
            i++; ws();
            if (peek('}')) { i++; return m; }
            while (true) {
                ws();
                if (!peek('"')) throw fail("expected a key in quotes");
                String k = string();
                ws();
                if (!peek(':')) throw fail("expected ':'");
                i++; ws();
                m.put(k, value(depth + 1));
                ws();
                if (peek(',')) { i++; continue; }
                if (peek('}')) { i++; return m; }
                throw fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            List<Object> l = new ArrayList<>();
            i++; ws();
            if (peek(']')) { i++; return l; }
            while (true) {
                ws();
                l.add(value(depth + 1));
                ws();
                if (peek(',')) { i++; continue; }
                if (peek(']')) { i++; return l; }
                throw fail("expected ',' or ']'");
            }
        }
        if (c == '"') return string();
        if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
        if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
        if (s.startsWith("null", i)) { i += 4; return null; }
        int st = i;
        while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
        if (st == i) throw fail("unexpected character '" + c + "'");
        try { return Double.valueOf(s.substring(st, i)); } catch (NumberFormatException e) { i = st; throw fail("bad number"); }
    }

    private boolean peek(char c) { return i < s.length() && s.charAt(i) == c; }

    private String string() {
        StringBuilder b = new StringBuilder();
        i++; // opening quote
        while (i < s.length()) {
            char c = s.charAt(i++);
            if (c == '"') return b.toString();
            if (c == '\\') {
                if (i >= s.length()) break;
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': b.append('\n'); break;
                    case 't': b.append('\t'); break;
                    case 'r': b.append('\r'); break;
                    case 'b': b.append('\b'); break;
                    case 'f': b.append('\f'); break;
                    case 'u':
                        if (i + 4 > s.length()) throw fail("bad \\u escape");
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; break;
                    default: b.append(e);
                }
            } else b.append(c);
        }
        throw fail("string is not closed");
    }

    /** Quotes and escapes a string for JSON output. */
    public static String quote(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : String.valueOf(s).toCharArray()) {
            if (c == '"' || c == '\\') b.append('\\').append(c);
            else if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
            else b.append(c);
        }
        return b.append('"').toString();
    }
}
