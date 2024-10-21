package ai.xpress.llama4s.model

import ai.xpress.llama4s.tensor.FloatTensor
import ai.xpress.llama4s.utils._
import scala.util.boundary
import java.util.random.RandomGenerator
import java.util.random.RandomGeneratorFactory
import boundary.break

type Sampler = FloatTensor => Int

object Sampler {
  def create(vocabSize: Int, temperature: Float, topp: Float, rngSeed: Long): Sampler = {
    if (temperature == 0.0f) {
      // Greedy argmax sampling: take the token with the highest probability
      _.argmax

    } else {
      val rng = RandomGeneratorFactory.getDefault.create(rngSeed)

      val inner: Sampler = {
        if (topp <= 0 || topp >= 1) {
          // Simply sample from the predicted probability distribution
          CategoricalSampler(rng)
        } else {
          // Top-P (nucleus) sampling, clamping the least likely tokens to zero
          // ToppSampler(vocabSize, topp, rng)
          ToppSampler(vocabSize, topp, rng)
        }
      }

      { (logits: FloatTensor) =>
        logits
          // Apply temperature to the logits
          ./=(0, logits.size, temperature)
          // Apply softmax to the logits to get the probabilities for next token
          .softmax_(0, logits.size)

        inner.apply(logits)
      }
    }
  }
}

final case class CategoricalSampler(rng: RandomGenerator) extends Sampler {
  def apply(logits: FloatTensor): Int = boundary {
    // Sample index from probabilities (they must sum to 1!)
    val random0to1 = rng.nextFloat(1f)
    var cdf = 0.0f
    for (i <- 0.until(logits.size)) {
      cdf += logits.get(i)
      if (random0to1 < cdf) {
        break(i)
      }
    }
    // In case of rounding errors
    logits.size - 1
  }
}

final case class ToppSampler(maxElements: Int, topp: Float, rng: RandomGenerator) extends Sampler {
  val indices = Array.ofDim[Int](maxElements)

  def apply(logits: FloatTensor): Int = {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    given Ordering[Int] = Ordering.map(logits.get).reverse

    val n = logits.size
    var head = 0
    var tail = n - 1
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    val cutoff: Float = (1.0f - topp) / (n - 1)
    for (i <- 0.until(indices.length)) {
      if (logits.get(i) >= cutoff) {
        indices(head) = i
        head += 1
      } else {
        indices(tail) = i
        tail -= 1
      }
    }

    val n0 = head
    // build heap O(n0)
    for (i <- (n0 / 2 - 1).to(0).by(-1)) {
      indices.siftDown(i, n0)
    }

    // Truncate the list where cumulative probability of the largest k elements exceeds topp
    // O(k lg n0)
    var cumulativeProb = 0.0f
    var lastIndex = 0
    boundary {
      for (i <- (n0 - 1).to(0).by(-1)) {
        indices.swap(0, i)
        cumulativeProb += logits.get(indices(i))

        if (cumulativeProb > topp) {
          lastIndex = i
          break() // we've exceeded topp by including lastIndex
        }

        indices.siftDown(0, i - 1)
      }
    }

    // sample from the truncated list
    val r = rng.nextFloat(1f) * cumulativeProb
    var cdf = 0.0f

    boundary {
      for (i <- (n0 - 1).to(lastIndex).by(-1)) {
        cdf = cdf + logits.get(indices(i))
        if (r < cdf) {
          break(i)
        }
      }

      indices(lastIndex) // in case of rounding errors
    }
  }
}
