package ai.xpress.llama4s.model

object ChatFormat {
  enum Role {
    case System,
      User,
      Assistant

    def name: String = {
      this.toString.toLowerCase
    }
  }

  final case class Message(role: Role, content: String)
}

final case class ChatFormat(tokenizer: Tokenizer) {
  private[model] val specialTokens = tokenizer.specialTokens

  val beginOfText = specialTokens("<|begin_of_text|>")
  val startHeader = specialTokens("<|start_header_id|>")
  val endHeader = specialTokens("<|end_header_id|>")
  val endOfTurn = specialTokens("<|eot_id|>")
  val endOfText = specialTokens("<|end_of_text|>")
  val endOfMessage = specialTokens.getOrElse("<|eom_id|>", -1) // only in 3.1
  val stopTokens = Set(endOfText, endOfTurn)

  def encodeHeader(role: ChatFormat.Role): Seq[Int] = {
    Seq.empty[Int].:+(startHeader).++(tokenizer.encode(role.name)).:+(endHeader).++(tokenizer.encode("\n"))
  }

  def encodeMessage(message: ChatFormat.Message): Seq[Int] = {
    encodeHeader(message.role).++(tokenizer.encode(message.content.strip)).:+(endOfTurn)
  }
}
