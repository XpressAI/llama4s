package ai.xpress.llama4s.tensor

import java.lang.foreign.MemorySegment
import java.lang.{Float => JFloat}
import jdk.incubator.vector.VectorSpecies

object ArrayFloatTensor {
  def allocate(dims: Int*)(using VectorSpecies[JFloat]): FloatTensor = {
    val size = FloatTensor.numElements(dims*)
    new ArrayFloatTensor(new Array[Float](size))
  }
}

final class ArrayFloatTensor(array: Array[Float])(using species: VectorSpecies[JFloat])
    extends F16FloatTensor(MemorySegment.ofArray(array), array.size)
