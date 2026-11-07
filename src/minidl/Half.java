package minidl;

/**
 * 16-bit number formats, written out by hand (so this works on any Java 17+):
 *   FP16  IEEE half precision  (1 sign, 5 exponent, 10 fraction bits)
 *   BF16  bfloat16             (1 sign, 8 exponent, 7 fraction bits = the top half of a float32)
 */
public final class Half {
    private Half() {}

    public static final float MAX_FP16 = 65504f;

    /** float -> FP16 bits, round to nearest even. Too big becomes infinity, too small becomes zero (or a subnormal). */
    public static short toFP16(float f) {
        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int e32 = (bits >>> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;
        if (e32 == 0xFF) return (short) (sign | 0x7C00 | (mant != 0 ? 0x200 : 0)); // infinity or NaN
        int exp = e32 - 127 + 15;
        if (exp >= 31) return (short) (sign | 0x7C00);                              // too big -> infinity
        if (exp <= 0) {                                                             // subnormal half, or zero
            if (exp < -10) return (short) sign;
            mant |= 0x800000;
            int shift = 14 - exp;
            int half = mant >> shift;
            int rem = mant & ((1 << shift) - 1), halfway = 1 << (shift - 1);
            if (rem > halfway || (rem == halfway && (half & 1) == 1)) half++;
            return (short) (sign | half);
        }
        int half = (exp << 10) | (mant >> 13);
        int rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1)) half++;             // a carry into the exponent is correct (it may become infinity)
        return (short) (sign | half);
    }

    public static float fromFP16(short h) {
        int sign = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        if (e == 0) {
            if (m == 0) return Float.intBitsToFloat(sign);
            float v = m * (1f / 16777216f); // subnormal: m * 2^-24
            return sign != 0 ? -v : v;
        }
        if (e == 31) return Float.intBitsToFloat(sign | 0x7F800000 | (m << 13));
        return Float.intBitsToFloat(sign | ((e + 112) << 23) | (m << 13));
    }

    /** float -> bfloat16 bits (the top 16 bits of the float, rounded to nearest even). */
    public static short toBF16(float f) {
        int bits = Float.floatToRawIntBits(f);
        return (short) ((bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16);
    }

    public static float fromBF16(short s) { return Float.intBitsToFloat(((int) s) << 16); }

    /** The value this number has after being stored in the given format. */
    public static float round(float f, Precision p) {
        switch (p) {
            case FP16: return fromFP16(toFP16(f));
            case BF16: return fromBF16(toBF16(f));
            default:   return f;
        }
    }

    public static void roundInPlace(float[] a, Precision p) {
        if (p == Precision.FP32) return;
        if (p == Precision.FP16) for (int i = 0; i < a.length; i++) a[i] = fromFP16(toFP16(a[i]));
        else for (int i = 0; i < a.length; i++) a[i] = fromBF16(toBF16(a[i]));
    }

    /** Stores a[from..from+n) as 16-bit numbers (2 bytes each). Reuses `reuse` when it has the right size. */
    public static short[] pack(float[] a, int from, int n, Precision p, short[] reuse) {
        short[] out = reuse != null && reuse.length == n ? reuse : new short[n];
        if (p == Precision.FP16) for (int i = 0; i < n; i++) out[i] = toFP16(a[from + i]);
        else for (int i = 0; i < n; i++) out[i] = toBF16(a[from + i]);
        return out;
    }

    public static void unpack(short[] s, int sFrom, float[] dst, int dFrom, int n, Precision p) {
        if (p == Precision.FP16) for (int i = 0; i < n; i++) dst[dFrom + i] = fromFP16(s[sFrom + i]);
        else for (int i = 0; i < n; i++) dst[dFrom + i] = fromBF16(s[sFrom + i]);
    }

    public static Tensor unpack(short[] s, int rows, int cols, Precision p) {
        Tensor t = new Tensor(rows, cols);
        unpack(s, 0, t.data, 0, rows * cols, p);
        return t;
    }
}
