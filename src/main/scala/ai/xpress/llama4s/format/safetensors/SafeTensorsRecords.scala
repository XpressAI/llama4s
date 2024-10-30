package ai.xpress.llama4s.format.safetensors

import ai.xpress.llama4s.tensor.DType
import ai.xpress.llama4s.tensor.ScalarDType

extension (info: STTensorInfo) {
  inline def validate: STTensorInfo = {
    assert(info.offsets.size == 2)
    info
  }

  inline def offset: Long = {
    info.offsets(0)
  }

  inline def sizeInBytes: Long = {
    info.offsets(1) - info.offsets(0)
  }

  inline def show: String = {
    f"${classOf[STTensorInfo].getSimpleName}: @[${info.offsets(0)}%-12s, ${info.offsets(1)}%-12s] ${info
        .dtype}%-8s ${info.shape.mkString("[", ", ", "]")}%-20s ${info.name}"
  }
}

extension (typ: STTensorInfo.DType) {
  inline def toDType: DType = {
    typ match {
      case STTensorInfo.DType.Bool =>
        ScalarDType.Bool
      case STTensorInfo.DType.I8 =>
        ScalarDType.I8
      case STTensorInfo.DType.U8 =>
        ScalarDType.U8
      case STTensorInfo.DType.I16 =>
        ScalarDType.I16
      case STTensorInfo.DType.U16 =>
        ScalarDType.U16
      case STTensorInfo.DType.F16 =>
        ScalarDType.F16
      case STTensorInfo.DType.Bf16 =>
        ScalarDType.BF16
      case STTensorInfo.DType.I32 =>
        ScalarDType.I32
      case STTensorInfo.DType.U32 =>
        ScalarDType.U32
      case STTensorInfo.DType.I64 =>
        ScalarDType.I64
      case STTensorInfo.DType.U64 =>
        ScalarDType.U64
      case STTensorInfo.DType.F64 =>
        ScalarDType.F64
      case other =>
        throw UnsupportedOperationException(f"Safetensors type currently unsupported: ${other}")
    }
  }
}
