package ai.xpress.llama4s.tensor

import ai.xpress.llama4s.gguf.Float16
import ai.xpress.llama4s.gguf.GGMLType
import ai.xpress.llama4s.utils.{_, given}
import java.lang.foreign.MemorySegment
import java.lang.{Byte => JByte}
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

final class Q4_0FloatTensor(buffer: MemorySegment, val size: Int)(using fspecies: VectorSpecies[JFloat])
    extends FloatTensor {
  override def set(index: Int, value: Float): Unit = {
    throw new UnsupportedOperationException("setFloat")
  }

  override def getVec(offset: Int): FloatVector = {
    throw new UnsupportedOperationException("getVec")
  }

  override val ggmlType: GGMLType = GGMLType.Q4_0

  override def get(index: Int): Float = {
    print(index)
    assert(0 <= index && index < size)

    // [ 2 byte scale factor | 4 bits per element x 32 scalar values ]
    val blockOffset = ggmlType.blockOffset(index)

    // Load the scale factor
    val scale = buffer.load[Short](blockOffset).f16ToF32

    // Load the quantized element
    val modIndex = index % ggmlType.blockSize
    val quant = {
      if (modIndex < ggmlType.blockSize / 2) {
        (buffer.load[Byte](blockOffset + Float16.BYTES + modIndex) & 0x0f).toByte
      } else {
        ((buffer.load[Byte](blockOffset + Float16.BYTES + modIndex - ggmlType.blockSize / 2) >>> 4) & 0x0f).toByte
      }
    }

    // Reconstruct the float
    (quant - 8) * scale
  }

  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    ggmlType.assertPowerOf2
    given VectorSpecies[JByte] = ByteVector.SPECIES_128
    given ByteOrder = ByteOrder.LITTLE_ENDIAN

    var result = 0f
    var j = 0

    // Align offset + j to ggmlType.blockSize
    val alignmentBound = Math.min(size, -offset & (ggmlType.blockSize - 1))
    if (alignmentBound > 0) {
      result += FloatTensor.dot(this, offset, other, otherOffset, alignmentBound)
      j += alignmentBound
    }

    // Ensure block alignment
    assert((offset + j) % ggmlType.blockSize == 0)

    // Create accumulator vector
    var accum = FloatVector.zero(fspecies)

    var blockOffset = ggmlType.blockOffset(offset + j)
    val upperBound = ggmlType.blockUpperBound(size)

    while (j < upperBound) {
      // Load the scale factor
      val scale = FloatVector.broadcast(fspecies, buffer.load[Short](blockOffset).f16ToF32)

      // Load the block
      val wBytes = buffer.loadBytesToVec(blockOffset + Float16.BYTES)
      val loBytes = wBytes.and(0xf.toByte).sub(8.toByte)
      val hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub(8.toByte)

      if (fspecies.vectorBitSize == 256) {
        // Perform elementwise-multiply
        val sum0 = other.getVec(otherOffset + j + 0 * fspecies.length) * loBytes.castTo[Float](0)
        val sum1 = other.getVec(otherOffset + j + 1 * fspecies.length) * loBytes.castTo[Float](1)
        val sum2 = other.getVec(otherOffset + j + 2 * fspecies.length) * hiBytes.castTo[Float](2)
        val sum3 = other.getVec(otherOffset + j + 3 * fspecies.length) * hiBytes.castTo[Float](3)
        // Add the elements
        accum = (sum0 + sum1 + sum2 + sum3).fma(scale, accum)

      } else if (fspecies.vectorBitSize == 128) {
        for (i <- 0 until 2) {
          val tmp = {
            if (i == 0) {
              loBytes
            } else {
              hiBytes
            }
          }

          val sum0 = other.getVec(otherOffset + j + (i * 4 + 0) * fspecies.length) * tmp.castTo[Float](0)
          val sum1 = other.getVec(otherOffset + j + (i * 4 + 1) * fspecies.length) * tmp.castTo[Float](1)
          val sum2 = other.getVec(otherOffset + j + (i * 4 + 2) * fspecies.length) * tmp.castTo[Float](2)
          val sum3 = other.getVec(otherOffset + j + (i * 4 + 3) * fspecies.length) * tmp.castTo[Float](3)
          accum = (sum0 + sum1 + sum2 + sum3).fma(scale, accum)
        }
      } else {
        throw new UnsupportedOperationException(fspecies.toString)
      }

      j += ggmlType.blockSize
      blockOffset += ggmlType.bytesPerBlock
    }

    result += accum.reduceLanes(VectorOperators.ADD)

    // Remaining entries
    if (j < size) {
      result += FloatTensor.dot(this, offset + j, other, otherOffset + j, size - j)
    }

    result
  }
}
