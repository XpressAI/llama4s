package ai.xpress.llama4s.tensor

import scala.util.Random
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import jdk.incubator.vector.VectorSpecies
import org.scalatest.matchers.should.Matchers._
import org.scalatest.wordspec.AnyWordSpec

final class F16FloatTensorUnitSpec extends AnyWordSpec {
  given ByteOrder = ByteOrder.LITTLE_ENDIAN
  given VectorSpecies[JFloat] = FloatTensor.FloatSpecies

  classOf[F16FloatTensor].getSimpleName should {
    val array = 0.to(Random.nextInt(50)).map(_ => Random.nextFloat).toArray
    val tensor = new ArrayFloatTensor(array)

    "correctly store Floats" in {
      tensor.toArray.toSeq should be (array.toSeq)
    }

    "perform dot product" in {
      tensor.dot(0, tensor, 0, tensor.size) should be (array.map(x => x * x).sum)
    }
  }
}
