package ai.xpress.llama4s.console

import ai.xpress.llama4s.model._
import ai.xpress.llama4s.tensor.FloatTensor
import scala.collection.mutable.{Buffer => MBuf}
import scala.util.boundary
import java.lang.{Float => JFloat}
import java.nio.ByteOrder
import java.util.Scanner
import jdk.incubator.vector.VectorSpecies
import boundary.break

final class LlamaConsole(model: LlamaModel, sampler: Sampler, options: ConsoleOptions)(using VectorSpecies[JFloat]) {
  def runInteractive: Unit = {
    val state = model.newState
    val format = ChatFormat(model.tokenizer)

    val conversation = MBuf(format.beginOfText)
    options
      .systemPrompt
      .foreach { sp =>
        conversation ++= format.encodeMessage(ChatFormat.Message(ChatFormat.Role.System, sp))
      }

    val scanner = new Scanner(System.in)
    boundary {
      while (true) {
        System.out.print("\n> ")
        System.out.flush
        val input = scanner.nextLine
        System.out.println("")

        if (Seq("quit", "exit").contains(input)) {
          System.out.println("Good bye!")
          break()
        }

        val start = conversation.size
        conversation ++= format.encodeMessage(ChatFormat.Message(ChatFormat.Role.User, input))
        conversation ++= format.encodeHeader(ChatFormat.Role.Assistant)

        // Generate the response
        var response = {
          model.generateTokens(
            state,
            start,
            conversation.drop(start).toSeq,
            format.stopTokens,
            options.maxTokens,
            sampler
          ) { token =>
            if (options.stream && !model.tokenizer.isSpecialToken(token)) {
              System.out.print(model.tokenizer.decode(token))
            }
          }
        }

        conversation ++= response

        var stopToken = Option.empty[Int]
        if (response.lastOption.map(format.stopTokens.contains(_)).getOrElse(false)) {
          stopToken = response.lastOption
          response = response.dropRight(1)
        }

        if (!options.stream) {
          System.out.println(model.tokenizer.decode(response))
        }

        if (stopToken.isEmpty) {
          System.err.println("\n\nRan out of context length...")
          break()
        }
      }
    }
  }

  def runSingleInstruction(input: String): Unit = {
    val state = model.newState
    val format = ChatFormat(model.tokenizer)

    // Create the prompt stream
    val prompt = MBuf(format.beginOfText)
    options
      .systemPrompt
      .foreach { sp =>
        prompt ++= format.encodeMessage(ChatFormat.Message(ChatFormat.Role.System, sp))
      }
    prompt ++= format.encodeMessage(ChatFormat.Message(ChatFormat.Role.User, input))
    prompt ++= format.encodeHeader(ChatFormat.Role.Assistant)

    // Generate the response
    var response = {
      model.generateTokens(state, 0, prompt.toSeq, format.stopTokens, options.maxTokens, sampler) { token =>
        if (options.stream && !model.tokenizer.isSpecialToken(token)) {
          System.out.print(model.tokenizer.decode(token))
        }
      }
    }

    if (response.lastOption.map(format.stopTokens.contains(_)).getOrElse(false)) {
      response = response.dropRight(1)
    }

    if (!options.stream) {
      System.out.println(model.tokenizer.decode(response))
    }
  }

  def run: Unit = {
    options.runMode match {
      case ConsoleOptions.RunMode.Interactive =>
        runInteractive
      case ConsoleOptions.RunMode.Once(input) =>
        runSingleInstruction(input)
    }
  }
}

object LlamaConsole {
  def main(args: Array[String]): Unit = {
    given ByteOrder = ByteOrder.LITTLE_ENDIAN
    given VectorSpecies[JFloat] = FloatTensor.FloatSpecies

    val options = ConsoleOptions.parser.constructOrExit(args)
    val model = LlamaModel.loadFromPath(options.modelPath, options.maxTokens, true)
    val sampler = Sampler.create(model.config.vocabularySize, options.temperature, options.topp, options.seed)
    val console = new LlamaConsole(model, sampler, options)

    console.run
  }
}
