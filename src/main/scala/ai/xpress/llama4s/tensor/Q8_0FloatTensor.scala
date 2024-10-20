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

final class Q8_0FloatTensor(buffer: MemorySegment, val size: Int)(using fspecies: VectorSpecies[JFloat])
    extends FloatTensor {
  override def set(index: Int, value: Float): Unit = {
    throw new UnsupportedOperationException("setFloat")
  }

  override def getVec(offset: Int): FloatVector = {
    throw new UnsupportedOperationException("getVec")
  }

  override val ggmlType: GGMLType = GGMLType.Q8_0

  override def get(index: Int): Float = {
    assert(0 <= index && index < size)

    // [ 2 byte scale factor | 8 bits per element x 32 scalar values ]
    val blockOffset = ggmlType.blockOffset(index)

    // Load the scale factor
    val scale = buffer.load[Short](blockOffset).f16ToF32

    // Load the quantized element
    val modIndex = index % ggmlType.blockSize
    val quant = buffer.load[Byte](blockOffset + Float16.BYTES + modIndex)

    // Reconstruct the float
    quant * scale
  }

  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    ggmlType.assertPowerOf2
    given bspecies: VectorSpecies[JByte] = {
      if (fspecies.vectorBitSize() == 256) {
        ByteVector.SPECIES_256
      } else {
        ByteVector.SPECIES_128
      }
    }
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

      if (fspecies.vectorBitSize == 256) {
        val bytes = buffer.loadBytesToVec(blockOffset + Float16.BYTES)
        val sum0 = other.getVec(otherOffset + j + 0 * fspecies.length) * bytes.castTo[Float](0)
        val sum1 = other.getVec(otherOffset + j + 1 * fspecies.length) * bytes.castTo[Float](1)
        val sum2 = other.getVec(otherOffset + j + 2 * fspecies.length) * bytes.castTo[Float](2)
        val sum3 = other.getVec(otherOffset + j + 3 * fspecies.length) * bytes.castTo[Float](3)
        accum = (sum0 + sum1 + sum2 + sum3).fma(scale, accum)

      } else if (fspecies.vectorBitSize == 128) {
        for (i <- 0 until 2) {
          val bytes = buffer.loadBytesToVec(blockOffset + Float16.BYTES + i * bspecies.vectorBitSize)
          val sum0 = other.getVec(otherOffset + j + i * 16 + 0 * fspecies.length) * bytes.castTo[Float](0)
          val sum1 = other.getVec(otherOffset + j + i * 16 + 1 * fspecies.length) * bytes.castTo[Float](1)
          val sum2 = other.getVec(otherOffset + j + i * 16 + 2 * fspecies.length) * bytes.castTo[Float](2)
          val sum3 = other.getVec(otherOffset + j + i * 16 + 3 * fspecies.length) * bytes.castTo[Float](3)
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
