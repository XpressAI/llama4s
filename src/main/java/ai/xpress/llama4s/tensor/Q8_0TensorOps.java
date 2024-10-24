package ai.xpress.llama4s.tensor;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.nio.ByteOrder;

interface Q8_0TensorOps {
    static final VectorSpecies<Float> fspecies = FloatTensor$.MODULE$.FloatSpecies();
    static final VectorSpecies<Byte> bspecies = (fspecies.vectorBitSize() == 256) ? ByteVector.SPECIES_256 : ByteVector.SPECIES_128;
    static final ByteOrder byteorder = ByteOrder.LITTLE_ENDIAN;

    static float vdot(FloatTensor thiz,
                            int thisOffset,
                            FloatTensor that,
                            int thatOffset,
                            int size) {
        var blockSize = thiz.ggmlType().blockSize();
        var bytesPerBlock = thiz.ggmlType().bytesPerBlock();

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

            if (fspecies.vectorBitSize() == 256) {
                var wBytes = ByteVector.fromMemorySegment(bspecies, thiz.buffer(), blockOffset + Short.BYTES, byteorder);
                var sum0 = that.getVec(thatOffset + j + 0 * fspecies.length()).mul(wBytes.castShape(fspecies, 0));
                var sum1 = that.getVec(thatOffset + j + 1 * fspecies.length()).mul(wBytes.castShape(fspecies, 1));
                var sum2 = that.getVec(thatOffset + j + 2 * fspecies.length()).mul(wBytes.castShape(fspecies, 2));
                var sum3 = that.getVec(thatOffset + j + 3 * fspecies.length()).mul(wBytes.castShape(fspecies, 3));
                accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum);

            } else if (fspecies.vectorBitSize() == 128) {
                // This loop cannot be unrolled, why?
                for (int i = 0; i < 2; ++i) {
                    var wBytes = ByteVector.fromMemorySegment(bspecies, thiz.buffer(), blockOffset + Short.BYTES + i * bspecies.vectorByteSize(), byteorder);
                    var sum0 = that.getVec(thatOffset + j + i * 16 + 0 * fspecies.length()).mul(wBytes.castShape(fspecies, 0));
                    var sum1 = that.getVec(thatOffset + j + i * 16 + 1 * fspecies.length()).mul(wBytes.castShape(fspecies, 1));
                    var sum2 = that.getVec(thatOffset + j + i * 16 + 2 * fspecies.length()).mul(wBytes.castShape(fspecies, 2));
                    var sum3 = that.getVec(thatOffset + j + i * 16 + 3 * fspecies.length()).mul(wBytes.castShape(fspecies, 3));
                    accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum);
                }

            } else {
                throw new UnsupportedOperationException(fspecies.toString());
            }
        }
        result += accum.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor$.MODULE$.dot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
