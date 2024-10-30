package ai.xpress.llama4s.model

import ai.xpress.llama4s.format.gguf.GGUFFile

object Vocabulary {
  def fromGGUF(file: GGUFFile): Vocabulary = {
    val model = file.header.metadata("tokenizer.ggml.model").asInstanceOf[String]
    if (model != Tokenizer.Llama3Model) {
      throw new IllegalArgumentException(s"Expected ${Tokenizer.Llama3Model} but found ${model}")
    }
    val tokens = file.header.metadata("tokenizer.ggml.tokens").asInstanceOf[Array[String]]
    Vocabulary(tokens)
  }
}

final case class Vocabulary(tokens: Array[String]) {
  override def toString: String = {
    s"${getClass.getSimpleName}(size = ${tokens.size})"
  }

  val tmap: Map[String, Int] = {
    tokens
      .zipWithIndex
      .map { (tok, i) =>
        (tok, i)
      }
      .toMap
  }

  def get(i: Int): String = {
    tokens(i)
  }

  def indexOf(token: String): Option[Int] = {
    tmap.get(token)
  }

  def size: Int = {
    tokens.size
  }
}
