package ai.xpress.llama4s.utils

import java.lang.foreign.MemorySegment
import java.lang.{Byte => JByte}
import java.nio.ByteOrder
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.VectorSpecies

opaque type MemLoad[T] = Long => T
opaque type MemStore[T] = (Long, T) => Unit

given _loadS: MemLoad[Short] = Unsafe.getShort
given _loadB: MemLoad[Byte] = Unsafe.getByte

given _storeS: MemStore[Short] = Unsafe.putShort
given _storeB: MemStore[Byte] = Unsafe.putByte

extension (buffer: MemorySegment) {
  def load[T <: AnyVal: MemLoad](offset: Long): T = {
    summon[MemLoad[T]](buffer.address + offset)
  }

  def store[T <: AnyVal: MemStore](offset: Long, value: T): MemorySegment = {
    summon[MemStore[T]](buffer.address + offset, value)
    buffer
  }

  def loadBytesToVec(offset: Long)(using species: VectorSpecies[JByte], bo: ByteOrder): ByteVector = {
    ByteVector.fromMemorySegment(species, buffer, offset, bo)
  }
}
