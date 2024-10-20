package ai.xpress.llama4s.gguf

import java.lang.{Byte => JByte}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}
import java.lang.{Short => JShort}

object Float16 {
  val BYTES = 2
}

val QK_K: Int = 256 // or 64?

enum GGMLType(val bytesPerBlock: Int, val blockSize: Int = 1) {
  case F32 extends GGMLType(JFloat.BYTES)
  case F16 extends GGMLType(Float16.BYTES)
  case Q4_0 extends GGMLType(Float16.BYTES + 16 * JByte.BYTES, 32)
  case Q4_1 extends GGMLType(2 * Float16.BYTES + 16 * JByte.BYTES, 32)
  case UNSUPPORTED_Q4_2 extends GGMLType(Int.MaxValue) // support has been removed
  case UNSUPPORTED_Q4_3 extends GGMLType(Int.MaxValue) // support has been removed
  case Q5_0 extends GGMLType(Int.MaxValue)
  case Q5_1 extends GGMLType(Int.MaxValue)
  case Q8_0 extends GGMLType(Float16.BYTES + 32 * JByte.BYTES, 32)
  case Q8_1 extends GGMLType(32 * JByte.BYTES + 2 * JFloat.BYTES, 32)
  // k-quantizations
  case Q2_K extends GGMLType(Int.MaxValue)
  case Q3_K extends GGMLType(Int.MaxValue)
  case Q4_K extends GGMLType(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 2, QK_K)
  case Q5_K extends GGMLType(2 * Float16.BYTES + ((QK_K / 16) / 8 * 6) + QK_K / 8 + QK_K / 2, QK_K)
  case Q6_K extends GGMLType(QK_K / 2 + QK_K / 4 + QK_K / 16 + Float16.BYTES, QK_K)
  case Q8_K extends GGMLType(Int.MaxValue)
  case I8 extends GGMLType(JByte.BYTES)
  case I16 extends GGMLType(JShort.BYTES)
  case I32 extends GGMLType(JInt.BYTES)

  inline def byteSizeFor(numElements: Int): Long = {
    val t = numElements.toLong * bytesPerBlock
    assert(t % blockSize == 0)
    t / blockSize
  }

  inline def blockOffset(index: Int): Int = {
    (index / blockSize) * bytesPerBlock
  }

  inline def assertPowerOf2: Unit = {
    assert(java.lang.Integer.bitCount(blockSize) == 1, "Power of 2")
  }

  inline def blockUpperBound(size: Int) = {
    size / blockSize * blockSize
  }
}
