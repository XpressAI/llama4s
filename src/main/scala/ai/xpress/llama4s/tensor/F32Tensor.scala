package ai.xpress.llama4s.tensor
import ai.xpress.llama4s.utils._
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

object F32Tensor {
  def allocate(dims: Int*)(using VectorSpecies[JFloat]): F32Tensor = {
    val size = FloatTensor.numElements(dims*)
    val segment = Unsafe.alloc(size * JFloat.BYTES)
    new F32Tensor(segment, size)
  }

  def from(array: Array[Float])(using VectorSpecies[JFloat]): F32Tensor = {
    new F32Tensor(MemorySegment.ofArray(array), array.size)
  }
}

class F32Tensor(val buffer: MemorySegment, val size: Int)(using species: VectorSpecies[JFloat]) extends FloatTensor {
  private[tensor] val layout = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

  override lazy val dtype: ScalarDType = ScalarDType.F32

  override def get(index: Int): Float = {
    buffer.get(layout, index.toLong * JFloat.BYTES)
  }

  override def set(index: Int, value: Float): Unit = {
    buffer.set(layout, index.toLong * JFloat.BYTES, value)
  }

  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    FloatTensor.dot(this, offset, other, otherOffset, size)
  }

  override def fill_(offset: Int, size: Int, value: Float): FloatTensor = {
    offset.until(offset + size).foreach(set(_, value))
    this
  }

  override def getVec(offset: Int): FloatVector = {
    FloatVector.fromMemorySegment(species, buffer, offset * JFloat.BYTES, ByteOrder.LITTLE_ENDIAN)
  }
}
