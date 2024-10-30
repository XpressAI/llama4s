package ai.xpress.llama4s.tensor;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.nio.ByteOrder;

interface Q4_0TensorOps {
    static final VectorSpecies<Float> fspecies = FloatTensor$.MODULE$.FloatSpecies();
    static final VectorSpecies<Byte> bspecies = ByteVector.SPECIES_128;
    static final ByteOrder byteorder = ByteOrder.LITTLE_ENDIAN;

    static float vdot(Q4_0Tensor thiz,
                        int thisOffset,
                        FloatTensor that,
                        int thatOffset,
                        int size) {
        var blockSize = thiz.dtype().blockSize();
        var bytesPerBlock = thiz.dtype().bytesPerBlock();

        var result = 0f;
        var j = 0;

        // Compute for the first few non-aligned entries
        int alignmentBound = Math.min(size, -thisOffset & (blockSize - 1));
        if (alignmentBound > 0) {
            result += FloatTensor$.MODULE$.dot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % blockSize == 0;

        var accum = FloatVector.zero(fspecies);
        int blockOffset = (thisOffset + j) / blockSize * bytesPerBlock;
        int upperBound = size / blockSize * blockSize;

        for (; j < upperBound; j += blockSize, blockOffset += bytesPerBlock) {
            // Load the scale factor
            var scale = FloatVector.broadcast(fspecies, FloatTensor$.MODULE$.loadF16(thiz.buffer(), blockOffset));

            // Load the block
            var wBytes = ByteVector.fromMemorySegment(bspecies, thiz.buffer(), blockOffset + Short.BYTES, byteorder);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            if (fspecies.vectorBitSize() == 256) {
                var sum0 = that.getVec(thatOffset + j + 0 * fspecies.length()).mul(loBytes.castShape(fspecies, 0));
                var sum1 = that.getVec(thatOffset + j + 1 * fspecies.length()).mul(loBytes.castShape(fspecies, 1));
                var sum2 = that.getVec(thatOffset + j + 2 * fspecies.length()).mul(hiBytes.castShape(fspecies, 0));
                var sum3 = that.getVec(thatOffset + j + 3 * fspecies.length()).mul(hiBytes.castShape(fspecies, 1));
                accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum);

            } else if (fspecies.vectorBitSize() == 128) {
                // This loop cannot be unrolled, why?
                for (int i = 0; i < 2; ++i) {
                    var tmp = i == 0 ? loBytes : hiBytes;
                    var sum0 = that.getVec(thatOffset + j + (i * 4 + 0) * fspecies.length()).mul(tmp.castShape(fspecies, 0));
                    var sum1 = that.getVec(thatOffset + j + (i * 4 + 1) * fspecies.length()).mul(tmp.castShape(fspecies, 1));
                    var sum2 = that.getVec(thatOffset + j + (i * 4 + 2) * fspecies.length()).mul(tmp.castShape(fspecies, 2));
                    var sum3 = that.getVec(thatOffset + j + (i * 4 + 3) * fspecies.length()).mul(tmp.castShape(fspecies, 3));
                    accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum);
                }

            } else {
                throw new UnsupportedOperationException(fspecies.toString());
            }
        }

        // Sum the accum elements
        result += accum.reduceLanes(VectorOperators.ADD);

        // Compute for the remaining entries
        if (j < size) {
            result += FloatTensor$.MODULE$.dot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
