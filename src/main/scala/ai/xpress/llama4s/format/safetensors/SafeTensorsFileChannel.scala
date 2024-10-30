package ai.xpress.llama4s.format.safetensors

import ai.xpress.llama4s.format.common.EnhancedFileChannel
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption

object SafeTensorsFileChannel {
  def from(filepath: Path)(using ByteOrder): SafeTensorsFileChannel = {
    new SafeTensorsFileChannel(FileChannel.open(filepath, StandardOpenOption.READ))
  }
}

final class SafeTensorsFileChannel private[safetensors] (val channel: FileChannel)(using byteorder: ByteOrder)
    extends EnhancedFileChannel[SafeTensorsFileChannel] {}
