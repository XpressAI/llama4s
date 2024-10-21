package ai.xpress.llama4s.utils

import java.lang.{Byte => JByte}
import java.lang.{Double => JDouble}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}
import java.lang.{Long => JLong}
import java.lang.{Short => JShort}
import java.nio.ByteBuffer
import java.nio.file.FileSystems
import java.nio.file.Files
import java.nio.file.Path
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.{Vector => JVector}

extension (vec: FloatVector) {
  inline def +(other: JVector[JFloat]): FloatVector = {
    vec.add(other)
  }

  inline def *(other: JVector[JFloat]): FloatVector = {
    vec.mul(other)
  }
}

type JNumeric = JByte | JShort | JFloat | JDouble | JInt | JLong

// format: off
type ToJavaPrimitive[T <: AnyVal | JNumeric] = T match {
  case JNumeric => T
  case Byte     => JByte
  case Short    => JShort
  case Float    => JFloat
  case Double   => JDouble
  case Int      => JInt
  case Long     => JLong
}
// format: on

extension [T](vec: JVector[T]) {
  inline def castTo[U <: AnyVal | JNumeric](
    part: Int
  )(using species: VectorSpecies[ToJavaPrimitive[U]]): JVector[ToJavaPrimitive[U]] = {
    vec.castShape(species, part)
  }
}

extension (x: Short) {
  inline def f16ToF32: Float = {
    JFloat.float16ToFloat(x)
  }
}

extension (x: Int) {
  inline def intBitsToFloat: Float = {
    JFloat.intBitsToFloat(x)
  }

  inline def bitCount: Int = {
    JInt.bitCount(x)
  }
}

extension (x: Long) {
  inline def longBitsToDouble: Double = {
    JDouble.longBitsToDouble(x)
  }

  inline def toIntExact: Int = {
    Math.toIntExact(x)
  }
}

extension (array: Array[Float]) {
  inline def toByteBuffer: ByteBuffer = {
    val buffer = ByteBuffer.allocate(array.size * JFloat.BYTES)
    buffer.asFloatBuffer.put(array)
    buffer
  }
}

extension (x: String) {
  inline def toPath: Path = {
    FileSystems.getDefault.getPath(x.replaceFirst("^~", System.getProperty("user.home"))).toAbsolutePath
  }
}

extension (path: Path) {
  inline def exists: Boolean = {
    Files.exists(path)
  }
}
