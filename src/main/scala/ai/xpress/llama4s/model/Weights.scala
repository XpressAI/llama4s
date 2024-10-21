package ai.xpress.llama4s.model

import ai.xpress.llama4s.gguf.GGMLTensorEntry
import ai.xpress.llama4s.tensor.ArrayFloatTensor
import ai.xpress.llama4s.tensor.FloatTensor
import java.lang.{Float => JFloat}
import jdk.incubator.vector.VectorSpecies

object Weights {
  private[model] def ropeFreqs(
      entries: Map[String, GGMLTensorEntry],
      config: LlamaConfig
  ): (Array[Float], Array[Float]) = {
    val ropeScaling = entries.contains("rope_freqs")
    val scaleFactor = 8f
    val loFreqFactor = 1f
    val hiFreqFactor = 3f
    val oldContextLength = 8192
    RoPE.precomputeFreqsCis(
      config.contextLength,
      config.headSize,
      config.ropeTheta,
      ropeScaling,
      scaleFactor,
      loFreqFactor,
      hiFreqFactor,
      oldContextLength
    )
  }

  def from(entries: Map[String, GGMLTensorEntry], config: LlamaConfig)(using
      species: VectorSpecies[JFloat]
  ): Weights = {
    val tokenEmbeddings = entries("token_embd.weight")
    val (ropeFreqsReal, ropeFreqsImag) = ropeFreqs(entries, config)

    Weights(
      tokenEmbeddings.asFloatT,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.attn_norm.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.attn_q.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.attn_k.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.attn_v.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.attn_output.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.ffn_norm.weight").asFloatT).toArray,
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.ffn_gate.weight").asFloatT).toArray, // w1
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.ffn_down.weight").asFloatT).toArray, // w2
      0.until(config.numberOfLayers).map(i => entries(s"blk.${i}.ffn_up.weight").asFloatT).toArray, // w3
      entries("output_norm.weight").asFloatT,
      new ArrayFloatTensor(ropeFreqsReal),
      new ArrayFloatTensor(ropeFreqsImag),
      // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
      // This is commonly referred as "tie word embeddings".
      entries.getOrElse("output.weight", tokenEmbeddings).asFloatT
    )
  }
}

final case class Weights(
    tokenEmbeddingTable: FloatTensor, // (vocab_size, dim)
    rmsAttWeight: Array[FloatTensor], // (layer, dim) rmsnorm weights
    wq: Array[FloatTensor], // (layer, n_heads * head_size)
    wk: Array[FloatTensor], // (layer, n_kv_heads, head_size)
    wv: Array[FloatTensor], // (layer, n_kv_heads * head_size)
    wo: Array[FloatTensor], // (layer, n_heads * head_size, dim)
    rmsFfnWeight: Array[FloatTensor], // (layer, dim)
    w1: Array[FloatTensor], // (layer, hidden_dim, dim)
    w2: Array[FloatTensor], // (layer, dim, hidden_dim)
    w3: Array[FloatTensor], // (layer, hidden_dim, dim)
    rmsFinalWeight: FloatTensor, // (dim,)
    freqCisReal: FloatTensor, // (seq_len, head_size/2)
    freqCisImag: FloatTensor, // (seq_len, head_size/2)
    wcls: FloatTensor // (vocab_size, dim)
)
