package ai.xpress.llama4s.model

import ai.xpress.llama4s.tensor.F32Tensor
import java.lang.{Float => JFloat}
import jdk.incubator.vector.VectorSpecies

final class LlamaState(config: LlamaConfig)(using VectorSpecies[JFloat]) {
  // current wave of activations

  // activation at current time stamp (dim,)
  val x = F32Tensor.allocate(config.dim)
  // same, but inside a residual branch (dim,)
  val xb = F32Tensor.allocate(config.dim)
  // an additional buffer just for convenience (dim,)
  val xb2 = F32Tensor.allocate(config.dim)
  // buffer for hidden dimension in the ffn (hidden_dim,)
  val hb = F32Tensor.allocate(config.hiddenDim)
  // buffer for hidden dimension in the ffn (hidden_dim,)
  val hb2 = F32Tensor.allocate(config.hiddenDim)
  // query (dim,)
  val q = F32Tensor.allocate(config.dim)
  // key (dim,)
  val k = F32Tensor.allocate(config.dim)
  // value (dim,)
  val v = F32Tensor.allocate(config.dim)
  // buffer for scores/attention values (n_heads, seq_len)
  val att = F32Tensor.allocate(config.numberOfHeads, config.contextLength)
  // output logits
  val logits = F32Tensor.allocate(config.vocabularySize)

  // KV cache
  val kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
  // (n_layer, seq_len, kv_dim)
  val keyCache = {
    1.to(config.numberOfLayers).map(_ => F32Tensor.allocate(config.contextLength, kvDim)).toArray
  }
  // (n_layer, seq_len, kv_dim)
  val valueCache = {
    1.to(config.numberOfLayers).map(_ => F32Tensor.allocate(config.contextLength, kvDim)).toArray
  }

  var latestToken = 0
}
