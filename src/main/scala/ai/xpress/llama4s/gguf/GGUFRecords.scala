package ai.xpress.llama4s.gguf

import ai.xpress.llama4s.gguf.GGMLType._
import ai.xpress.llama4s.tensor._
import ai.xpress.llama4s.utils._
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import jdk.incubator.vector.VectorSpecies

final case class GGUFTensorInfo private[gguf] (name: String, dimensions: Array[Int], ggmlType: GGMLType, offset: Long)

final case class GGMLTensorEntry private[gguf] (
    name: String,
    ggmlType: GGMLType,
    shape: Array[Int],
    mappedFile: MemorySegment,
    segment: MemorySegment
) {
  def asFloatT(using VectorSpecies[JFloat]): FloatTensor = {
    ggmlType match {
      case Q8_0 =>
        Q8_0FloatTensor(segment, FloatTensor.numElements(shape*))
      case Q4_0 =>
        Q4_0FloatTensor(segment, FloatTensor.numElements(shape*))
      case F32 =>
        F16FloatTensor(segment, FloatTensor.numElements(shape*))
      case other =>
        throw UnsupportedOperationException(f"GGML type currently unsupported: ${other}")
    }
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

object GGUFFile {
  def from(path: Path)(using ByteOrder): GGUFFile = {
    from(GGUFFileChannel.from(path))
  }

  def from(channel: GGUFFileChannel)(using ByteOrder): GGUFFile = {
    // Reset the channel position
    channel.position(0)

    // gguf_header_t: The header of the file
    val header = channel.readHeader

    // gguf_tensor_info_t: tensor infos, which can be used to locate the tensor data.
    val tensorInfos = {
      0.until(header.tensorCount).map(_ => channel.readTensorInfo(header.alignment)).map(x => (x.name, x)).toMap
    }

    // Padding to the nearest multiple of `ALIGNMENT`.
    // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
    // long _padding = -fileChannel.position() & (ALIGNMENT - 1);
    val padding = header.alignment - (channel.position % header.alignment)
    channel.position(channel.position + padding)

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    // uint8_t tensor_data[];
    val tensorDataOffset = channel.position

    GGUFFile(header, tensorInfos, tensorDataOffset)
  }
}

final case class GGUFFile(header: GGUFHeader, tensorInfos: Map[String, GGUFTensorInfo], tensorDataOffset: Long) {
  def loadTensors(channel: FileChannel): Map[String, GGMLTensorEntry] = {
    // Reset position
    channel.position(0)

    // mmap the file
    val arena = Arena.ofAuto
    val mapped = channel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, channel.size - tensorDataOffset, arena)

    tensorInfos.map { (_, info) =>
      val numElems = FloatTensor.numElements(info.dimensions*)
      val sizeInBytes = info.ggmlType.byteSizeFor(numElems).toIntExact
      val buffer = mapped.asSlice(info.offset, sizeInBytes)
      (info.name, GGMLTensorEntry(info.name, info.ggmlType, info.dimensions, mapped, buffer))
    }
  }
}
