package ai.xpress.llama4s.model

import ai.xpress.llama4s.tensor.ArrayFloatTensor
import java.lang.{Float => JFloat}
import jdk.incubator.vector.VectorSpecies

final class LlamaState(config: LlamaConfig)(using VectorSpecies[JFloat]) {
  // current wave of activations

  // activation at current time stamp (dim,)
  val x = ArrayFloatTensor.allocate(config.dim)
  // same, but inside a residual branch (dim,)
  val xb = ArrayFloatTensor.allocate(config.dim)
  // an additional buffer just for convenience (dim,)
  val xb2 = ArrayFloatTensor.allocate(config.dim)
  // buffer for hidden dimension in the ffn (hidden_dim,)
  val hb = ArrayFloatTensor.allocate(config.hiddenDim)
  // buffer for hidden dimension in the ffn (hidden_dim,)
  val hb2 = ArrayFloatTensor.allocate(config.hiddenDim)
  // query (dim,)
  val q = ArrayFloatTensor.allocate(config.dim)
  // key (dim,)
  val k = ArrayFloatTensor.allocate(config.dim)
  // value (dim,)
  val v = ArrayFloatTensor.allocate(config.dim)
  // buffer for scores/attention values (n_heads, seq_len)
  val att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength)
  // output logits
  val logits = ArrayFloatTensor.allocate(config.vocabularySize)

  // KV cache
  val kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
  // (n_layer, seq_len, kv_dim)
  val keyCache = {
    1.to(config.numberOfLayers).map(_ => ArrayFloatTensor.allocate(config.contextLength, kvDim)).toArray
  }
  // (n_layer, seq_len, kv_dim)
  val valueCache = {
    1.to(config.numberOfLayers).map(_ => ArrayFloatTensor.allocate(config.contextLength, kvDim)).toArray
  }

  var latestToken = 0
}
