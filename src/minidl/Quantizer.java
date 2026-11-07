package minidl;

import java.util.ArrayList;
import java.util.List;

/**
 * Shrinks a trained model for prediction. Three flavours:
 *   quantize(model)            Dense + Conv2D weights -> int8   (4x smaller than float32)
 *   quantizeInt4(model, group) Dense weights -> int4 (8x smaller), Conv2D -> int8
 *   toBF16(model)              Dense weights -> bfloat16 (2x smaller)
 * Layers inside Residual / TokenWise blocks are converted too. Other layers are shared with the original, still float.
 * The returned model can predict but not train.
 */
public final class Quantizer {
    private Quantizer() {}

    private enum Mode { INT8, INT4, BF16 }

    public static Sequential quantize(Sequential model) { return convertAll(model, Mode.INT8, 0); }
    public static Sequential quantizeInt4(Sequential model, int group) { return convertAll(model, Mode.INT4, group); }
    public static Sequential toBF16(Sequential model) { return convertAll(model, Mode.BF16, 0); }

    private static Sequential convertAll(Sequential model, Mode mode, int group) {
        Sequential q = new Sequential();
        for (Layer l : model.layers()) q.add(convert(l, mode, group));
        q.setTraining(false);
        return q;
    }

    private static Layer convert(Layer l, Mode mode, int group) {
        if (l instanceof Dense) {
            Dense d = (Dense) l;
            switch (mode) {
                case INT4: return QuantizedDense4.from(d, group);
                case BF16: return BF16Dense.from(d);
                default:   return QuantizedDense.from(d);
            }
        }
        if (l instanceof Conv2D && mode != Mode.BF16 && ((Conv2D) l).isPlain()) return QuantizedConv2D.from((Conv2D) l);
        if (l instanceof Residual) return new Residual(convertList(((Residual) l).inner(), mode, group));
        if (l instanceof TokenWise) {
            TokenWise t = (TokenWise) l;
            return new TokenWise(t.steps(), t.inDim(), convertList(t.inner(), mode, group));
        }
        return l;
    }

    private static List<Layer> convertList(List<Layer> in, Mode mode, int group) {
        List<Layer> out = new ArrayList<>();
        for (Layer l : in) out.add(convert(l, mode, group));
        return out;
    }
}
