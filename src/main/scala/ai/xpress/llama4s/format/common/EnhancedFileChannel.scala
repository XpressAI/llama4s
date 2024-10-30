package ai.xpress.llama4s.format.common

import ai.xpress.llama4s.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets

private[format] trait EnhancedFileChannel[T](using byteorder: ByteOrder) {
  val BB_1 = ByteBuffer.allocate(1).order(byteorder)
  val BB_2 = ByteBuffer.allocate(2).order(byteorder)
  val BB_4 = ByteBuffer.allocate(4).order(byteorder)
  val BB_8 = ByteBuffer.allocate(8).order(byteorder)

  def channel: FileChannel

  inline def position: Long = {
    channel.position
  }

  inline def position(p: Long): EnhancedFileChannel[T] = {
    channel.position(p)
    this
  }

  inline def readByte: Byte = {
    val bytesRead = channel.read(BB_1)
    assert(bytesRead == 1)
    BB_1.clear.get(0)
  }

  inline def readBoolean: Boolean = {
    readByte != 0
  }

  inline def readShort: Short = {
    val bytesRead = channel.read(BB_2)
    assert(bytesRead == 2)
    BB_2.clear.getShort(0)
  }

  inline def readInt: Int = {
    val bytesRead = channel.read(BB_4)
    assert(bytesRead == 4)
    BB_4.clear.getInt(0)
  }

  inline def readLong: Long = {
    val bytesRead = channel.read(BB_8)
    assert(bytesRead == 8)
    BB_8.clear.getLong(0)
  }

  inline def readFloat: Float = {
    readInt.intBitsToFloat
  }

  inline def readDouble: Double = {
    readLong.longBitsToDouble
  }

  inline def readString: String = {
    // Read the length of the string
    val len = readLong.toIntExact

    // The string as a UTF-8 non-null-terminated string.
    val bytes = new Array[Byte](len)
    val bytesRead = channel.read(ByteBuffer.wrap(bytes))

    assert(len == bytesRead)
    new String(bytes, StandardCharsets.UTF_8)
  }

  inline def readJson: JValue = {
    JsonMethods.parse(readString)
  }
}
