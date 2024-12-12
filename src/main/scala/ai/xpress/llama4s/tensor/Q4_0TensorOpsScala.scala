package ai.xpress.llama4s.tensor

import java.lang.{Float => JFloat, Byte => JByte, Short => JShort}
import jdk.incubator.vector.{ByteVector, FloatVector, VectorOperators, VectorSpecies}
import java.nio.ByteOrder

object Q4_0TensorOpsScala {
  val fspecies: VectorSpecies[JFloat] = FloatTensor.FloatSpecies
  val bspecies: VectorSpecies[JByte] = ByteVector.SPECIES_128
  val byteorder: ByteOrder = ByteOrder.LITTLE_ENDIAN

  def vdot(thiz: Q4_0Tensor, thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): JFloat = {
    val blockSize = thiz.dtype.blockSize
    val bytesPerBlock = thiz.dtype.bytesPerBlock

    var result = 0f
    var j = 0

    // Compute for the first few non-aligned entries
    val alignmentBound = Math.min(size, -thisOffset & (blockSize - 1))
    if (alignmentBound > 0) {
      result += FloatTensor.dot(thiz, thisOffset, that, thatOffset, alignmentBound)
      j += alignmentBound
    }
    assert((thisOffset + j) % blockSize == 0)

    var accum = FloatVector.zero(fspecies)
    var blockOffset = (thisOffset + j) / blockSize * bytesPerBlock
    val upperBound = size / blockSize * blockSize

    while (j < upperBound) {
      // Load the scale factor
      val scale = FloatVector.broadcast(fspecies, FloatTensor.loadF16(thiz.buffer, blockOffset))

      // Load the block
      val wBytes = ByteVector.fromMemorySegment(bspecies, thiz.buffer, blockOffset + JShort.BYTES, byteorder)
      val loBytes = wBytes.and(0xF.toByte).sub(8.toByte)
      val hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub(8.toByte)

      fspecies.vectorBitSize() match {
        case 256 =>
          val sum0 = that.getVec(thatOffset + j + 0 * fspecies.length()).mul(loBytes.castShape(fspecies, 0))
          val sum1 = that.getVec(thatOffset + j + 1 * fspecies.length()).mul(loBytes.castShape(fspecies, 1))
          val sum2 = that.getVec(thatOffset + j + 2 * fspecies.length()).mul(hiBytes.castShape(fspecies, 0))
          val sum3 = that.getVec(thatOffset + j + 3 * fspecies.length()).mul(hiBytes.castShape(fspecies, 1))
          accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum)

        case 128 =>
          for (i <- 0 until 2) {
            val tmp = if (i == 0) loBytes else hiBytes
            val sum0 = that.getVec(thatOffset + j + (i * 4 + 0) * fspecies.length()).mul(tmp.castShape(fspecies, 0))
            val sum1 = that.getVec(thatOffset + j + (i * 4 + 1) * fspecies.length()).mul(tmp.castShape(fspecies, 1))
            val sum2 = that.getVec(thatOffset + j + (i * 4 + 2) * fspecies.length()).mul(tmp.castShape(fspecies, 2))
            val sum3 = that.getVec(thatOffset + j + (i * 4 + 3) * fspecies.length()).mul(tmp.castShape(fspecies, 3))
            accum = sum0.add(sum1).add(sum2).add(sum3).fma(scale, accum)
          }

        case _ =>
          throw new UnsupportedOperationException(fspecies.toString)
      }

      j += blockSize
      blockOffset += bytesPerBlock
    }

    // Sum the accum elements
    result += accum.reduceLanes(VectorOperators.ADD)

    // Compute for the remaining entries
    if (j < size) {
      result += FloatTensor.dot(thiz, thisOffset + j, that, thatOffset + j, size - j)
    }

    result
  }
}

