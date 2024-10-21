package ai.xpress.llama4s.gguf

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
