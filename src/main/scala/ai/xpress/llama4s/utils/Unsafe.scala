package ai.xpress.llama4s.utils

import sun.misc.{Unsafe => JUnsafe}
import java.lang.foreign.MemorySegment
import java.lang.{Integer => JInt}
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

val Unsafe = {
  try {
    val f = classOf[JUnsafe].getDeclaredField("theUnsafe")
    f.setAccessible(true)
    f.get(null).asInstanceOf[JUnsafe]
  } catch {
    case exc: (NoSuchFieldException | IllegalAccessException) =>
      throw new RuntimeException(exc)
  }
}

extension (unsafe: JUnsafe) {
  def CacheLineSize: Int = 64

  def addressOffset: Long = {
    unsafe.objectFieldOffset(classOf[Buffer].getDeclaredField("address"))
  }

  def alloc(capacity: Int, align: Int = unsafe.CacheLineSize): MemorySegment = {
    if (capacity <= 0) {
      throw new IllegalArgumentException("Capacity must be > 0");
    }

    if (JInt.bitCount(align) != 1) {
      throw new IllegalArgumentException("Alignment must be a power of 2");
    }

    val buffer = ByteBuffer.allocateDirect(capacity + align)
    val address = unsafe.getLong(buffer, addressOffset)

    if ((address & (align - 1)) == 0) {
      buffer.limit(capacity)

    } else {
      val pos = (align - (address & (align - 1))).toInt
      buffer.position(pos)
      buffer.limit(pos + capacity)
    }

    MemorySegment.ofBuffer(buffer.slice.order(ByteOrder.nativeOrder))
  }
}
