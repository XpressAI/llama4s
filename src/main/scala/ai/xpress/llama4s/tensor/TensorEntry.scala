package ai.xpress.llama4s.tensor

import java.lang.foreign.MemorySegment
import java.lang.{Float => JFloat}
import jdk.incubator.vector.VectorSpecies

final case class TensorEntry(name: String, dtype: DType, shape: Array[Int], segment: MemorySegment) {
  inline def sizeInBytes: Long = {
    dtype.byteSizeFor(FloatTensor.numElements(shape*))
  }

  inline def show: String = {
    f"${getClass.getSimpleName}: ${dtype}%-10s ${shape.mkString("[ ", ", ", " ]")}%-20s ${name}"
  }

  def asFloatT(using VectorSpecies[JFloat]): FloatTensor = {
    dtype match {
      case QuantizedDType.Q8_0 =>
        Q8_0Tensor(segment, FloatTensor.numElements(shape*))
      case QuantizedDType.Q4_0 =>
        Q4_0Tensor(segment, FloatTensor.numElements(shape*))
      case ScalarDType.F32 =>
        F32Tensor(segment, FloatTensor.numElements(shape*))
      case other =>
        throw UnsupportedOperationException(f"Currently unsupported dtype: ${other}")
    }
  }
}
