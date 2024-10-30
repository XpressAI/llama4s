package ai.xpress.llama4s.tensor

import java.lang.Enum
import java.lang.{Byte => JByte}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}

object Float16 {
  val BYTES = 2
}

type DType = ScalarDType | QuantizedDType

extension (dtype: DType) {
  inline def byteSizeFor(numElements: Int): Long = {
    dtype match {
      case typ: ScalarDType =>
        typ.byteSizeFor(numElements)
      case typ: QuantizedDType =>
        typ.byteSizeFor(numElements)
    }
  }
}

enum ScalarDType(bytes: Int) extends Enum[QuantizedDType] {
  case Bool extends ScalarDType(1)
  case I8 extends ScalarDType(1)
  case U8 extends ScalarDType(1)
  case I16 extends ScalarDType(2)
  case U16 extends ScalarDType(2)
  case F16 extends ScalarDType(2)
  case BF16 extends ScalarDType(2)
  case I32 extends ScalarDType(4)
  case U32 extends ScalarDType(4)
  case F32 extends ScalarDType(4)
  case I64 extends ScalarDType(8)
  case U64 extends ScalarDType(8)
  case F64 extends ScalarDType(8)

  inline def byteSizeFor(numElements: Int): Long = {
    numElements.toLong * bytes.toLong
  }
}

private[tensor] val QK_K: Int = 256 // or 64?

enum QuantizedDType(val bytesPerBlock: Int, val blockSize: Int = 1) extends Enum[QuantizedDType] {
  // Q4
  case Q4_0 extends QuantizedDType(Float16.BYTES + 16 * JByte.BYTES, 32)
  case Q4_1 extends QuantizedDType(2 * Float16.BYTES + 16 * JByte.BYTES, 32)
  // Q5
  case Q5_0 extends QuantizedDType(Int.MaxValue)
  case Q5_1 extends QuantizedDType(Int.MaxValue)
  // Q8
  case Q8_0 extends QuantizedDType(Float16.BYTES + 32 * JByte.BYTES, 32)
  case Q8_1 extends QuantizedDType(32 * JByte.BYTES + 2 * JFloat.BYTES, 32)
  // K-quantizations
  case Q2_K extends QuantizedDType(Int.MaxValue)
  case Q3_K extends QuantizedDType(Int.MaxValue)
  case Q4_K extends QuantizedDType(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 2, QK_K)
  case Q5_K extends QuantizedDType(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 8 + QK_K / 2, QK_K)
  case Q6_K extends QuantizedDType(QK_K / 2 + QK_K / 4 + QK_K / 16 + Float16.BYTES, QK_K)
  case Q8_K extends QuantizedDType(Int.MaxValue)

  inline def byteSizeFor(numElements: Int): Long = {
    val t = numElements.toLong * bytesPerBlock
    assert(t % blockSize == 0)
    t / blockSize
  }

  inline def blockOffset(index: Int): Int = {
    (index / blockSize) * bytesPerBlock
  }

  inline def assertPowerOf2: Unit = {
    assert(JInt.bitCount(blockSize) == 1, "Power of 2")
  }

  inline def blockUpperBound(size: Int) = {
    size / blockSize * blockSize
  }
}
