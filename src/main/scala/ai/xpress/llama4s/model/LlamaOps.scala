package ai.xpress.llama4s.model

import ai.xpress.llama4s.tensor.FloatTensor
import ai.xpress.llama4s.utils._
import scala.collection.mutable.{Buffer => MBuf}
import scala.util.boundary
import java.util.stream.IntStream
import boundary.break

object LlamaOps {
  def rmsnorm(out: FloatTensor, x: FloatTensor, weight: FloatTensor, size: Int, rmsNormEps: Float): Unit = {
    // calculate sum of squares
    var ss = {
      x.foldLeft(0, size, 0f) { case (acc, xi) =>
        acc + xi * xi
      }
    }
    ss /= size
    ss += rmsNormEps
    ss = (1.0 / Math.sqrt(ss)).asInstanceOf[Float]

    // normalize and scale
    val finalss = ss // for the lambda
    out.mapWithIndex_(0, size) { (value, index) =>
      weight.get(index) * (finalss * x.get(index))
    }
  }
}

extension (model: LlamaModel) {
  def forward(state: LlamaState, position: Int): FloatTensor = {
    val token = state.latestToken
    val config = model.config
    val weights = model.weights

    val dim = config.dim
    val headSize = config.headSize
    val kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
    val kvMul = {
      config.numberOfHeads / config.numberOfKeyValueHeads // integer multiplier of the kv sharing in multiquery
    }
    val sqrtHeadSize = Math.sqrt(headSize).toFloat

    // copy the token embedding into x
    weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim)

    // Forward all the layers
    for (l <- 0 until config.numberOfLayers) {
      // Attention rmsnorm
      LlamaOps.rmsnorm(state.xb, state.x, weights.rmsAttWeight(l), dim, config.rmsNormEps)

      // qkv matmuls for this position
      weights.wq(l).matmul_(state.xb, state.q, dim, dim)
      weights.wk(l).matmul_(state.xb, state.k, kvDim, dim)
      weights.wv(l).matmul_(state.xb, state.v, kvDim, dim)

      // RoPE relative positional encoding: complex-valued rotate q and k in each head
      for (i <- 0.until(dim).by(2)) {
        val head_dim = i % headSize
        val fcr = weights.freqCisReal.get(position * (headSize / 2) + (head_dim / 2))
        val fci = weights.freqCisImag.get(position * (headSize / 2) + (head_dim / 2))
        val rotn = {
          if (i < kvDim) {
            2
          } else {
            1 // how many vectors? 2 = q & k, 1 = q only
          }
        }
        for (v <- 0 until rotn) {
          val vec = {
            if (v == 0) {
              state.q
            } else {
              state.k // the vector to rotate (query or key)
            }
          }
          val (v0, v1) = (vec.get(i), vec.get(i + 1))
          vec.set(i, v0 * fcr - v1 * fci)
          vec.set(i + 1, v0 * fci + v1 * fcr)
        }
      }

      // save key,value at this time step (position) to our kv cache
      // int loff = l * config.seq_len * kvDim // kv cache layer offset for convenience
      state.k.copyTo(0, state.keyCache(l), position * kvDim, kvDim)
      state.v.copyTo(0, state.valueCache(l), position * kvDim, kvDim)

      val curLayer = l

      // multihead attention. iterate over all heads
      IntStream
        .range(0, config.numberOfHeads)
        .parallel
        .forEach { h =>
          // get the query vector for this head
          // float* q = s.q + h * headSize
          val qOffset = h * headSize

          // attention scores for this head
          // float* att = s.att + h * config.seq_len
          val attOffset = h * config.contextLength

          // iterate over all timesteps, including the current one
          for (t <- 0 to position) {
            // get the key vector for this head and at this timestep
            // float* k = s.key_cache + loff + t * dim + h * headSize
            val keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize
            // calculate the attention score as the dot product of q and k
            val score = state.q.dot(qOffset, state.keyCache(curLayer), keyCacheOffset, headSize) / sqrtHeadSize
            // save the score to the attention buffer
            state.att.set(attOffset + t, score)
          }

          // softmax the scores to get attention weights, from 0..position inclusively
          state.att.softmax_(attOffset, position + 1)

          // weighted sum of the values, store back into xb
          // float* xb = s.xb + h * headSize
          val xbOffset = h * headSize
          // memset(xb, 0, headSize * sizeof(float))
          state.xb.fill_(xbOffset, headSize, 0f)

          for (t <- 0 to position) {
            // get the value vector for this head and at this timestep
            // float* v = s.value_cache + loff + t * dim + h * headSize
            val vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize
            // get the attention weight for this timestep
            val a = state.att.get(attOffset + t)
            // accumulate the weighted value into xb
            state.xb.saxpy_(xbOffset, state.valueCache(curLayer), vOffset, headSize, a)
          }
        }

      // final matmul to get the output of the attention
      weights.wo(l).matmul_(state.xb, state.xb2, dim, dim)

      // residual connection back into x
      state.x += state.xb2

      // ffn rmsnorm
      LlamaOps.rmsnorm(state.xb, state.x, weights.rmsFfnWeight(l), dim, config.rmsNormEps)

      // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
      // first calculate self.w1(x) and self.w3(x)
      weights.w1(l).matmul_(state.xb, state.hb, config.hiddenDim, dim)
      weights.w3(l).matmul_(state.xb, state.hb2, config.hiddenDim, dim)

      // SwiGLU non-linearity
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      state.hb.map_(x => x / (1.0 + Math.exp(-x)).toFloat)

      // elementwise multiply with w3(x)
      state.hb *= state.hb2

      // final matmul to get the output of the ffn
      weights.w2(l).matmul_(state.hb, state.xb, dim, config.hiddenDim)

      // residual connection
      state.x += state.xb
    }

    // final rmsnorm
    LlamaOps.rmsnorm(state.x, state.x, weights.rmsFinalWeight, dim, config.rmsNormEps)

    // classifier into logits
    weights.wcls.matmul_(state.x, state.logits, config.vocabularySize, dim)

    state.logits
  }

  def generateTokens(
      state: LlamaState,
      startPos: Int,
      prompt: Seq[Int],
      stopTokens: Set[Int],
      _maxTokens: Int,
      sampler: Sampler
  )(callback: Int => Unit): Seq[Int] = {
    val maxTokens = {
      if (_maxTokens < 0 || _maxTokens > model.config.contextLength) {
        model.config.contextLength
      } else {
        _maxTokens
      }
    }
    var nextToken = 0

    val promptIter = prompt.iterator
    val response = MBuf.empty[Int]

    boundary {
      for (position <- startPos until maxTokens) {
        forward(state, position)

        if (promptIter.hasNext) {
          nextToken = promptIter.next
          // System.err.print(model.tokenizer.decode(nextToken).replaceControlChars)

        } else {
          nextToken = sampler.apply(state.logits)
          // System.err.print(model.tokenizer.decode(nextToken).replaceControlChars)

          response += nextToken
          callback(nextToken)
          if (stopTokens.contains(nextToken)) {
            break()
          }
        }

        state.latestToken = nextToken
      }
    }

    response.toSeq
  }
}
