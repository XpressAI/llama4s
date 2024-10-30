package ai.xpress.llama4s.format.gguf

import ai.xpress.llama4s.format.gguf.GGMLType._
import ai.xpress.llama4s.tensor._
import ai.xpress.llama4s.utils._

final case class GGUFTensorInfo private[gguf] (name: String, shape: Array[Int], ggmlType: GGMLType, offset: Long) {
  inline def sizeInBytes: Long = {
    ggmlType.toDType.byteSizeFor(FloatTensor.numElements(shape*))
  }

  inline def show: String = {
    f"${getClass.getSimpleName}: @${offset}%-12s ${ggmlType}%-10s ${shape.mkString("[", ", ", "]")}%-20s ${name}"
  }
}

object GGUFHeader {
  val MagicValue = 0x46554747
  val SupportedGGUFVersions = Seq(2, 3)
  val DefaultAlignment = 32 // must be a power of 2
}

final case class GGUFHeader private[gguf] (version: Int, tensorCount: Int, metadata: Map[String, AnyVal | AnyRef]) {
  lazy val alignment: Int = {
    val avalue = metadata.getOrElse("general.alignment", GGUFHeader.DefaultAlignment).asInstanceOf[Int]

    assert(avalue.bitCount == 1, "alignment must be a power of two")
    avalue
  }
}

enum MetadataValueType(val byteSize: Int) {
  // The value is a 8-bit unsigned integer.
  case UINT8 extends MetadataValueType(1)
  // The value is a 8-bit signed integer.
  case INT8 extends MetadataValueType(1)
  // The value is a 16-bit unsigned little-endian integer.
  case UINT16 extends MetadataValueType(2)
  // The value is a 16-bit signed little-endian integer.
  case INT16 extends MetadataValueType(2)
  // The value is a 32-bit unsigned little-endian integer.
  case UINT32 extends MetadataValueType(4)
  // The value is a 32-bit signed little-endian integer.
  case INT32 extends MetadataValueType(4)
  // The value is a 32-bit IEEE754 floating point number.
  case FLOAT32 extends MetadataValueType(4)
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
  case BOOL extends MetadataValueType(1)
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  case STRING extends MetadataValueType(-8)
  // The value is an array of other values, with the length and type prepended.
  // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
  case ARRAY extends MetadataValueType(-8)
  // The value is a 64-bit unsigned little-endian integer.
  case UINT64 extends MetadataValueType(8)
  // The value is a 64-bit signed little-endian integer.
  case INT64 extends MetadataValueType(8)
  // The value is a 64-bit IEEE754 floating point number.
  case FLOAT64 extends MetadataValueType(8)
}
