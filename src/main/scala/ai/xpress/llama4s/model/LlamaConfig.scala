package ai.xpress.llama4s.model

import ai.xpress.llama4s.gguf.GGUFFile

object LlamaConfig {
  def fromGGUF(file: GGUFFile): LlamaConfig = {
    val metadata = file.header.metadata
    LlamaConfig(
      metadata("llama.embedding_length").asInstanceOf[Int],
      metadata("llama.feed_forward_length").asInstanceOf[Int],
      metadata("llama.block_count").asInstanceOf[Int],
      metadata("llama.attention.head_count").asInstanceOf[Int],
      metadata.getOrElse("llama.attention.head_count_kv", metadata("llama.attention.head_count")).asInstanceOf[Int],
      0, // Fill in later
      metadata("llama.context_length").asInstanceOf[Int],
      metadata.getOrElse("llama.attention.layer_norm_rms_epsilon", 1e-5f).asInstanceOf[Float],
      metadata.getOrElse("llama.rope.freq_base", 10000f).asInstanceOf[Float]
    )
  }
}

final case class LlamaConfig(
    dim: Int, // transformer dimension
    hiddenDim: Int, // for ffn layers
    numberOfLayers: Int, // number of layers
    numberOfHeads: Int, // number of query heads
    numberOfKeyValueHeads: Int, // number of key/value heads (can be < query heads because of multiquery)
    vocabularySize: Int, // vocabulary size, usually 256 (byte-level)
    contextLength: Int, // max sequence length
    rmsNormEps: Float,
    ropeTheta: Float
) {
  val headSize: Int = dim / numberOfHeads
}
