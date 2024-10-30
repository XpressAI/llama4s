package ai.xpress.llama4s.format.safetensors

import ai.xpress.llama4s.tensor.DType
import ai.xpress.llama4s.tensor.ScalarDType
import ai.xpress.llama4s.tensor.TensorEntry
import ai.xpress.llama4s.utils._
import org.json4s.JsonDSL._
import org.json4s._
import scalapb.json4s.JsonFormat
import java.lang.foreign.Arena
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path

object SafeTensorsFile {
  val DATA = "data"
  val METADATA = "__metadata__"

  def from(path: Path)(using ByteOrder): SafeTensorsFile = {
    from(SafeTensorsFileChannel.from(path))
  }

  def from(channel: SafeTensorsFileChannel): SafeTensorsFile = {
    // Reset the channel position
    channel.position(0)

    // Read the JSON in the front
    val json = channel.readJson

    // Remove the `__metadata__` key, wrap inside another JSON, and parse into
    // protobuf to extract the tensor infos
    val tensorInfos = JsonFormat
      .fromJson[STTensorInfoList](DATA -> (json -- METADATA))
      .data
      .map { (name, info) =>
        (name, info.copy(name = name).validate)
      }

    // Extract the `__metadata__` key and parse as Map[String, Any]
    val metadata = (json \ METADATA).asInstanceOf[JObject].values

    SafeTensorsFile(metadata, tensorInfos, channel.position)
  }
}

final case class SafeTensorsFile(
    metadata: Map[String, Any],
    tensorInfos: Map[String, STTensorInfo],
    tensorDataOffset: Long
) {
  def show: String = {
    val builder = StringBuilder()
    builder ++= "\n\nMetadata:\n"
    metadata.foreach { (k, v) =>
      builder ++= f"  ${k}%-20s : ${v}\n"
    }
    builder ++= "\n" ++= f"Tensors @ ${tensorDataOffset}\n"
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
        TensorEntry(info.name, info.dtype.toDType, info.shape, segment)
      }
      .toMap
  }
}
