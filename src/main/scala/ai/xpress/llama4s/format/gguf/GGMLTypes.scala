package ai.xpress.llama4s.format.gguf

import ai.xpress.llama4s.tensor.DType
import ai.xpress.llama4s.tensor.QuantizedDType
import ai.xpress.llama4s.tensor.ScalarDType

enum GGMLType {
  case F32
  case F16
  case Q4_0
  case Q4_1
  case UNSUPPORTED_Q4_2
  case UNSUPPORTED_Q4_3
  case Q5_0
  case Q5_1
  case Q8_0
  case Q8_1
  case Q2_K
  case Q3_K
  case Q4_K
  case Q5_K
  case Q6_K
  case Q8_K
  case I8
  case I16
  case I32

  inline def toDType: DType = {
    this match {
      case I8 =>
        ScalarDType.I8
      case I16 =>
        ScalarDType.I16
      case I32 =>
        ScalarDType.I32
      case F32 =>
        ScalarDType.F32
      case F16 =>
        ScalarDType.F16
      case Q4_0 =>
        QuantizedDType.Q4_0
      case Q4_1 =>
        QuantizedDType.Q4_1
      case Q5_0 =>
        QuantizedDType.Q5_0
      case Q5_1 =>
        QuantizedDType.Q5_1
      case Q8_0 =>
        QuantizedDType.Q8_0
      case Q8_1 =>
        QuantizedDType.Q8_1
      case Q2_K =>
        QuantizedDType.Q2_K
      case Q3_K =>
        QuantizedDType.Q3_K
      case Q4_K =>
        QuantizedDType.Q4_K
      case Q5_K =>
        QuantizedDType.Q5_K
      case Q6_K =>
        QuantizedDType.Q6_K
      case Q8_K =>
        QuantizedDType.Q8_K
      case other =>
        throw UnsupportedOperationException(f"GGML type currently unsupported: ${other}")
    }
  }
}
