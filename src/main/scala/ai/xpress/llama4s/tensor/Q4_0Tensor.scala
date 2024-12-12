package ai.xpress.llama4s.tensor

import ai.xpress.llama4s.utils.{_, given}
import java.lang.foreign.MemorySegment
import java.lang.{Float => JFloat, Byte => JByte, Short => JShort}
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.ByteVector
import java.nio.ByteOrder

final class Q4_0Tensor(val buffer: MemorySegment, val size: Int)(using fspecies: VectorSpecies[JFloat])
    extends FloatTensor {

  override def set(index: Int, value: Float): Unit = {
    throw new UnsupportedOperationException("setFloat")
  }

  override def getVec(offset: Int): FloatVector = {
    throw new UnsupportedOperationException("getVec")
  }

  override lazy val dtype: QuantizedDType = QuantizedDType.Q4_0

  override def get(index: Int): Float = {
    assert(0 <= index && index < size)

    // [ 2 byte scale factor | 4 bits per element x 32 scalar values ]
    val blockOffset = dtype.blockOffset(index)

    // Load the scale factor
    val scale = buffer.load[Short](blockOffset).f16ToF32

    // Load the quantized element
    val modIndex = index % dtype.blockSize
    val quant = {
      if (modIndex < dtype.blockSize / 2) {
        (buffer.load[Byte](blockOffset + Float16.BYTES + modIndex) & 0x0f).toByte
      } else {
        ((buffer.load[Byte](blockOffset + Float16.BYTES + modIndex - dtype.blockSize / 2) >>> 4) & 0x0f).toByte
      }
    }

    // Reconstruct the float
    (quant - 8) * scale
  }

  val bspecies: VectorSpecies[JByte] = ByteVector.SPECIES_128
  val byteorder: ByteOrder = ByteOrder.LITTLE_ENDIAN


  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    Q4_0TensorOps.vdot(this, offset, other.asInstanceOf[FloatTensor], otherOffset, size)
    // Q4_0TensorOpsScala.vdot(this, offset, other.asInstanceOf[FloatTensor], otherOffset, size)
    // vdot2(offset, other.asInstanceOf[FloatTensor], otherOffset, size)
  }

  def vdot2(thisOffset: Int, that: FloatTensor, thatOffset: Int, size: Int): JFloat = {
    val blockSize = dtype.blockSize
    val bytesPerBlock = dtype.bytesPerBlock

    var result = 0f
    var j = 0

    // Compute for the first few non-aligned entries
    val alignmentBound = Math.min(size, -thisOffset & (blockSize - 1))
    if (alignmentBound > 0) {
      result += FloatTensor.dot(this, thisOffset, that, thatOffset, alignmentBound)
      j += alignmentBound
    }
    assert((thisOffset + j) % blockSize == 0)

    var accum = FloatVector.zero(fspecies)
    var blockOffset = (thisOffset + j) / blockSize * bytesPerBlock
    val upperBound = size / blockSize * blockSize

    while (j < upperBound) {
      // Load the scale factor
      val scale = FloatVector.broadcast(fspecies, FloatTensor.loadF16(buffer, blockOffset))

      // Load the block
      val wBytes = ByteVector.fromMemorySegment(bspecies, buffer, blockOffset + JShort.BYTES, byteorder)
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
      result += FloatTensor.dot(this, thisOffset + j, that, thatOffset + j, size - j)
    }

    result
  }

}
