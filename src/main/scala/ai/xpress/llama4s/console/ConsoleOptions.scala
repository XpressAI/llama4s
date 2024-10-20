package ai.xpress.llama4s.console

import ai.xpress.llama4s.utils._
import mainargs.Flag
import mainargs.ParserForClass
import mainargs.TokensReader
import mainargs.arg
import mainargs.main
import java.nio.file.Path

object ConsoleOptions {
  given TokensReader.Simple[Path] with {
    def shortName = "path"
    def read(strs: Seq[String]): Either[String, Path] = {
      strs.toList match {
        case p :: _ =>
          val path = p.toPath
          if (path.exists) {
            Right(path)
          } else {
            Left(s"Path ${path} does not exist!")
          }
        case _ =>
          Left("No path provided!")
      }
    }
  }

  val parser = ParserForClass[ConsoleOptions]
}

final case class ConsoleOptions(
    @arg(name = "model", short = 'm', doc = "Path to the model file (.gguf)")
    modelPath: Path,
    @arg(short = 'i', doc = "Run in interactive chat mode")
    interactive: Flag,
    @arg(short = 'p', doc = "Input prompt")
    prompt: String,
    @arg(doc = "System prompt")
    systemPrompt: Option[String],
    @arg(doc = "Temperature, range: [0, inf] (default: 0.1)")
    temperature: Float = 0.1f,
    @arg(doc = "P value in Top-P (nucleus) sampling, range: [0, 1] (default: 0.95)")
    topp: Float = 0.95f,
    @arg(doc = "Random seed (default: System.nanoTime)")
    seed: Long = System.nanoTime,
    @arg(doc = "Number of steps to run for < 0 = limited by context length (default: 512)")
    maxTokens: Int = 512,
    @arg(doc = "Print tokens during generation (default: true)")
    stream: Flag = Flag(true),
    @arg(doc = "Print ALL tokens to stderr")
    echo: Flag = Flag(false)
)
