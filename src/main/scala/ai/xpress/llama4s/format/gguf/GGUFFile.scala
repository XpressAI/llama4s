package ai.xpress.llama4s.format.gguf

import ai.xpress.llama4s.format.gguf.GGMLType._
import ai.xpress.llama4s.tensor._
import java.lang.foreign.Arena
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path

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
    // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)]
    // long _padding = -fileChannel.position() & (ALIGNMENT - 1)
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
    // uint8_t tensor_data[]
    val tensorDataOffset = channel.position

    GGUFFile(header, tensorInfos, tensorDataOffset)
  }
}

final case class GGUFFile(header: GGUFHeader, tensorInfos: Map[String, GGUFTensorInfo], tensorDataOffset: Long) {
  def show: String = {
    val builder = StringBuilder()
    builder ++= f"\nTensors @ ${tensorDataOffset}\n"
    tensorInfos.foreach { (_, info) =>
      builder ++= info.show + "\n"
    }
    builder.toString
  }

  def loadTensors(channel: FileChannel): Map[String, TensorEntry] = {
    // Reset position
    channel.position(0)

    // mmap the file
    val arena = Arena.ofAuto
    val mmapped = channel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, channel.size - tensorDataOffset, arena)

    tensorInfos
      .mapValues { info =>
        val segment = mmapped.asSlice(info.offset, info.sizeInBytes)
        TensorEntry(info.name, info.ggmlType.toDType, info.shape, segment)
      }
      .toMap
  }
}
