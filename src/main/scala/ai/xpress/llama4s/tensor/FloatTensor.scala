package ai.xpress.llama4s.tensor

import ai.xpress.llama4s.gguf.GGMLType
import ai.xpress.llama4s.utils.{_, given}
import java.lang.Boolean
import java.lang.foreign.MemorySegment
import java.util.stream.IntStream
import jdk.incubator.vector.FloatVector

object FloatTensor {
  val UseVectorApi = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"))

  // Preferred vector size for the fast multiplication routines.
  // (Apple Silicon) NEON only supports up-to 128bit vectors.
  val FloatSpecies = {
    if (FloatVector.SPECIES_PREFERRED.vectorBitSize == 128) {
      FloatVector.SPECIES_128
    } else {
      FloatVector.SPECIES_256
    }
  }

  def loadF16(buffer: MemorySegment, offset: Int): Float = {
    buffer.load[Short](offset).f16ToF32
  }

  inline def numElements(dims: Int*): Int = {
    assert(dims.size > 0 && dims.forall(_ > 0))
    dims.reduce(_ * _)
  }

  def dot(x: FloatTensor, xOffset: Int, y: FloatTensor, yOffset: Int, size: Int): Float = {
    var accum = 0f
    for (j <- 0 until size) {
      accum += (x.get(xOffset + j) * y.get(yOffset + j))
    }
    accum
  }
}

trait FloatTensor {
  override def toString: String = {
    val pos = {
      if (size < 5) {
        size
      } else {
        5
      }
    }
    val ed = {
      if (size < 5) {
        " ]"
      } else {
        ", ... ]"
      }
    }
    s"${getClass.getSimpleName}(size = ${size}, ${0.until(pos).map(get(_)).mkString("[ ", ", ", ed)} )"
  }

  def buffer: MemorySegment

  def size: Int

  def get(index: Int): Float

  def set(index: Int, value: Float): Unit

  def getVec(offset: Int): FloatVector

  def toArray(offset: Int, size: Int): Array[Float] = {
    offset.until(offset + size).map(get(_)).toArray
  }

  def toArray: Array[Float] = {
    toArray(0, size)
  }

  def ggmlType: GGMLType

  def dot(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Float

  def matmul_(other: FloatTensor, out: FloatTensor, dim0: Int, dim1: Int): Unit = {
    IntStream.range(0, dim0).parallel.forEach(i => out.set(i, dot(i * dim1, other, 0, dim1)))
  }

  def map_(offset: Int, size: Int)(func: Float => Float): FloatTensor = {
    offset
      .until(offset + size)
      .foreach { i =>
        set(i, func(get(i)))
      }
    this
  }

  def mapWithIndex_(offset: Int, size: Int)(func: (Float, Int) => Float): FloatTensor = {
    offset
      .until(offset + size)
      .foreach { i =>
        set(i, func(get(i), i))
      }
    this
  }

  def map_(func: Float => Float): FloatTensor = {
    return map_(0, size)(func)
  }

  def foldLeft(offset: Int, size: Int, seed: Float)(func: (Float, Float) => Float): Float = {
    var accum = seed
    for (i <- 0 until size) {
      accum = func(accum, get(offset + i))
    }
    accum
  }

  def copyTo(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): Unit = {
    other.mapWithIndex_(otherOffset, size) { (_, index) =>
      this.get(index - otherOffset + offset)
    }
  }

  def add_(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): FloatTensor = {
    mapWithIndex_(offset, size) { (value, index) =>
      value + other.get(index - offset + otherOffset)
    }
  }

  def +=(other: FloatTensor): FloatTensor = {
    return add_(0, other, 0, size)
  }

  def mul_(offset: Int, other: FloatTensor, otherOffset: Int, size: Int): FloatTensor = {
    mapWithIndex_(offset, size) { (value, index) =>
      value * other.get(index - offset + otherOffset)
    }
  }

  def *=(other: FloatTensor): FloatTensor = {
    return mul_(0, other, 0, size)
  }

  def /=(offset: Int, size: Int, value: Float): FloatTensor = {
    map_(offset, size)(_ / value)
  }

  def fill_(offset: Int, size: Int, value: Float): FloatTensor = {
    map_(offset, size)(_ => value)
  }

  def softmax_(offset: Int, size: Int): FloatTensor = {
    // Find max value (for numerical stability)
    val maxval = max(offset, size)
    // Exp and sum
    map_(offset, size) { x =>
      Math.exp(x - maxval).asInstanceOf[Float]
    }
    val sumval = sum(offset, size)
    // Normalize
    /=(offset, size, sumval)
  }

  def sum(offset: Int, size: Int): Float = {
    foldLeft(offset, size, 0f)(_ + _)
  }

  def max(offset: Int, size: Int): Float = {
    foldLeft(offset, size, Float.NegativeInfinity)(Math.max(_, _))
  }

  def argmax(offset: Int, size: Int): Int = {
    assert(size > 0)
    var maxIndex = offset
    var maxValue = get(maxIndex)
    val endIndex = offset + size
    for (i <- offset until endIndex) {
      val f = get(i)
      if (f > maxValue) {
        maxValue = f
        maxIndex = i
      }
    }
    maxIndex
  }

  def argmax: Int = {
    argmax(0, size)
  }

  def saxpy_(offset: Int, other: FloatTensor, otherOffset: Int, size: Int, a: Float): FloatTensor = {
    // this[otherOffset ... otherOffset + size) = a * other[otherOffset ... otherOffset + size) + this[offset ... offset + size)
    for (i <- 0 until size) {
      set(offset + i, a * other.get(otherOffset + i) + get(offset + i))
    }
    this
  }
}
