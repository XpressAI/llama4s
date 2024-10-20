package ai.xpress.llama4s.tensor

import ai.xpress.llama4s.gguf.GGMLType
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

class F16FloatTensor(segment: MemorySegment, val size: Int)(using species: VectorSpecies[JFloat]) extends FloatTensor {
  private[tensor] val layout = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

  override def get(index: Int): Float = {
    segment.get(layout, index.toLong * JFloat.BYTES)
  }

  override def set(index: Int, value: Float): Unit = {
    segment.set(layout, index.toLong * JFloat.BYTES, value)
  }

  override def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float = {
    FloatTensor.dot(this, offset, other, otherOffset, size)
  }

  override val ggmlType: GGMLType = GGMLType.F32

  override def fill_(offset: Int, size: Int, value: Float): FloatTensor = {
    offset.until(offset + size).foreach(set(_, value))
    this
  }

  override def getVec(offset: Int): FloatVector = {
    FloatVector.fromMemorySegment(species, segment, offset * JFloat.BYTES, ByteOrder.nativeOrder)
  }
}
