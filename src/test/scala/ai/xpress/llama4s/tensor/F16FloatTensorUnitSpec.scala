package ai.xpress.llama4s.tensor

import scala.util.Random
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import jdk.incubator.vector.VectorSpecies
import org.scalatest.matchers.should.Matchers._
import org.scalatest.wordspec.AnyWordSpec

final class F32TensorUnitSpec extends AnyWordSpec {
  given ByteOrder = ByteOrder.LITTLE_ENDIAN
  given VectorSpecies[JFloat] = FloatTensor.FloatSpecies

  classOf[F32Tensor].getSimpleName should {
    "correctly store Floats" in {
      val array = 0.to(Random.nextInt(50)).map(_ => Random.nextFloat).toArray
      val tensor = F32Tensor.from(array)
      tensor.toArray.toSeq should be(array.toSeq)
    }

    "corectly perform dot product" in {
      val size = Random.nextInt(50)
      val a1 = 0.to(size).map(_ => Random.nextFloat).toArray
      val t1 = F32Tensor.from(a1)
      val a2 = 0.to(size).map(_ => Random.nextFloat).toArray
      val t2 = F32Tensor.from(a2)

      t1.dot(0, t2, 0, t1.size) should be(a1.zip(a2).map(_ * _).sum)
    }
  }
}
