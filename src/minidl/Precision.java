package minidl;

/** Number formats for mixed-precision training. */
public enum Precision {
    /** 32-bit float: 8 exponent bits, 23 fraction bits. The default. */
    FP32,
    /** IEEE half: 5 exponent bits, 10 fraction bits. Precise, but small range (6e-8 .. 65504), so it needs loss scaling. */
    FP16,
    /** bfloat16: 8 exponent bits (same range as FP32), only 7 fraction bits. Needs no loss scaling. */
    BF16
}
