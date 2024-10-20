package ai.xpress.llama4s.model

import ai.xpress.llama4s.gguf._
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import java.nio.file.Path
import jdk.incubator.vector.VectorSpecies

object LlamaModel {
  def loadFromPath(path: Path, contextLength: Int, loadWeights: Boolean)(using
      ByteOrder,
      VectorSpecies[JFloat]
  ): LlamaModel = {
    val channel = GGUFFileChannel.from(path)
    val file = GGUFFile.from(channel)

    val vocabulary = Vocabulary.fromGGUF(file)
    val tokenizer = Tokenizer.fromGGUF(file, vocabulary)
    val config = LlamaConfig.fromGGUF(file).copy(vocabularySize = vocabulary.size, contextLength = contextLength)
    val entries = file.loadTensors(channel.channel)
    val weights = Weights.from(entries, config)

    LlamaModel(config, tokenizer, weights)
  }
}

final case class LlamaModel(config: LlamaConfig, tokenizer: Tokenizer, weights: Weights) {
  def newState(using VectorSpecies[JFloat]): LlamaState = {
    val state = new LlamaState(config)
    state.latestToken = tokenizer.specialTokens("<|begin_of_text|>")
    state
  }
}
