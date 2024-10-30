package ai.xpress.llama4s.tensor

import ai.xpress.llama4s.utils.{_, given}
import java.lang.foreign.MemorySegment
import java.lang.{Float => JFloat}
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

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

  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    Q4_0TensorOps.vdot(this, offset, other.asInstanceOf[FloatTensor], otherOffset, size)
  }
}
