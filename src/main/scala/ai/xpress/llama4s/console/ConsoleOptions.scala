package ai.xpress.llama4s.console

import ai.xpress.llama4s.utils._
import mainargs.ParserForClass
import mainargs.TokensReader
import mainargs.arg
import mainargs.main
import java.nio.file.Path

object ConsoleOptions {
  enum RunMode {
    case Interactive
    case Once(input: String)
  }

  given TokensReader.Simple[RunMode] with {
    def shortName = "single"
    def read(input: Seq[String]): Either[String, RunMode] = {
      input.toList match {
        case prompt :: _ =>
          Right(RunMode.Once(prompt))
        case _ =>
          Left("No prompt provided!")
      }
    }
  }

  given TokensReader.Simple[Path] with {
    def shortName = "path"
    def read(input: Seq[String]): Either[String, Path] = {
      input.toList match {
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
    @arg(name = "prompt", short = 'p', doc = "Run in single-prompt mode")
    runMode: ConsoleOptions.RunMode = ConsoleOptions.RunMode.Interactive,
    @arg(doc = "System prompt")
    systemPrompt: Option[String],
    @arg(doc = "Temperature, range: [0, inf] (default: 0.1)")
    temperature: Float = 0.1f,
    @arg(doc = "P value in Top-P (nucleus) sampling, range: [0, 1] (default: 0.95)")
    topp: Float = 0.95f,
    @arg(doc = "Random seed (default: System.nanoTime)")
    seed: Long = System.nanoTime,
    @arg(doc = "Number of steps to run for < 0 = limited by context length (default: 16384)")
    maxTokens: Int = 16384,
    @arg(doc = "Print tokens during generation (default: true)")
    stream: Boolean = true,
    @arg(doc = "Print ALL tokens to stderr")
    echo: Boolean = false
)
