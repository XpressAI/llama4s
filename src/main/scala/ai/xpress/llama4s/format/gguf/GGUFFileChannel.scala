package ai.xpress.llama4s.format.gguf

import ai.xpress.llama4s.format.common.EnhancedFileChannel
import ai.xpress.llama4s.utils._
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import MetadataValueType._

object GGUFFileChannel {
  def from(filepath: Path)(using ByteOrder): GGUFFileChannel = {
    new GGUFFileChannel(FileChannel.open(filepath, StandardOpenOption.READ))
  }
}

final class GGUFFileChannel private[gguf] (val channel: FileChannel)(using byteorder: ByteOrder)
    extends EnhancedFileChannel[GGUFFileChannel] {

  private[gguf] def readMetadataValueType: MetadataValueType = {
    // gguf_metadata_value_type
    MetadataValueType.fromOrdinal(readInt)
  }

  private[gguf] def readGGMLType: GGMLType = {
    // ggml_type
    GGMLType.fromOrdinal(readInt)
  }

  private[gguf] def readArray: AnyRef = {
    // Any value type is valid, including arrays
    // gguf_metadata_value_type
    val valtype = readMetadataValueType

    // Number of elements, not bytes
    val len = readLong.toIntExact

    // The array of values.
    // gguf_metadata_value_t array[len]
    valtype match {
      case UINT8 | INT8 =>
        0.until(len).map(_ => readByte).toArray
      case UINT16 | INT16 =>
        0.until(len).map(_ => readShort).toArray
      case UINT32 | INT32 =>
        0.until(len).map(_ => readInt).toArray
      case UINT64 | INT64 =>
        0.until(len).map(_ => readLong).toArray
      case FLOAT32 =>
        0.until(len).map(_ => readFloat).toArray
      case FLOAT64 =>
        0.until(len).map(_ => readDouble).toArray
      case BOOL =>
        0.until(len).map(_ => readBoolean).toArray
      case STRING =>
        0.until(len).map(_ => readString).toArray
      case ARRAY =>
        0.until(len).map(_ => readArray).toArray
    }
  }

  private[gguf] def readMetadataValue: AnyVal | AnyRef = {
    readMetadataValueType match {
      // gguf_metadata_value_t
      case UINT8 | INT8 =>
        readByte
      case UINT16 | INT16 =>
        readShort
      case UINT32 | INT32 =>
        readInt
      case UINT64 | INT64 =>
        readLong
      case FLOAT32 =>
        readFloat
      case FLOAT64 =>
        readDouble
      case BOOL =>
        readBoolean
      case STRING =>
        readString
      case ARRAY =>
        readArray
    }
  }

  private[gguf] def readKVPair: (String, AnyVal | AnyRef) = {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.

    // gguf_string_t key
    val key = readString
    assert(key.length < (1 << 16))
    assert(
      key
        .codePoints
        .allMatch { cp =>
          ('a' <= cp && cp <= 'z') ||
          ('0' <= cp && cp <= '9') || cp == '_' || cp == '.'
        }
    )
    (key, readMetadataValue)
  }

  def readHeader: GGUFHeader = {
    // uint32_t: Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    val magic = readInt
    if (magic != GGUFHeader.MagicValue) {
      throw new IllegalArgumentException(s"unsupported header.magic value: ${magic}")
    }

    // uint32_t: The version of the format implemented.
    // Must be `3` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update
    // the metadata to signify the change.
    val version = readInt
    if (!GGUFHeader.SupportedGGUFVersions.contains(version)) {
      throw new IllegalArgumentException("unsupported header.version " + version)
    }

    // uint64_t: The number of tensors in the file
    // This is explicit, instead of being included in the metadata, to ensure it
    // is always present for loading the tensors.
    val tensorCount = readLong.toIntExact
    // uint64_t: The number of metadata key-value pairs
    val kvCount = readLong.toIntExact
    // gguf_metadata_kv_t: The metadata key-value pairs
    val metadata = 0.until(kvCount).map(_ => readKVPair).toMap
    GGUFHeader(version, tensorCount, metadata)
  }

  private[gguf] def readTensorInfo(alignment: Int): GGUFTensorInfo = {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    // gguf_string_t name
    val name = readString
    assert(name.length <= 64)

    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    // uint32_t
    val ndims = readInt
    assert(ndims <= 4)
    // The dimensions of the tensor
    // uint64_t
    val dimensions = 0.until(ndims).map(_ => readLong.toIntExact).toArray

    // The type of the tensor.
    // ggml_type
    val ggmlType = readGGMLType

    // The offset of the tensor's data in this file in bytes.
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    // Must be a multiple of `ALIGNMENT`.
    // uint64_t
    val offset = readLong
    assert(offset % alignment == 0)
    GGUFTensorInfo(name, dimensions, ggmlType, offset)
  }
}
