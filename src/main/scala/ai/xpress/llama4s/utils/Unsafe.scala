package ai.xpress.llama4s.utils

import sun.misc.{Unsafe => JUnsafe}

opaque type Token = Int

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
