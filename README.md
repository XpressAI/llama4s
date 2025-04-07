# Llama4S

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Scala Version](https://img.shields.io/badge/Scala-3.5.1-red.svg)](https://www.scala-lang.org/)
[![Discord](https://img.shields.io/badge/Discord-Join%20us!-7289DA?logo=discord)](https://discord.com/invite/vgEg2ZtxCw)

Practical [Llama 3](https://github.com/meta-llama/llama3), [3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1) and [3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) inference implemented purely in Scala 3, leveraging the Java Vector API for performance.

This project supports running Llama models in GGUF format.

## Prerequisites

*   Java Development Kit (JDK) 21 or later (required for the Vector API).
*   [sbt](https://www.scala-sbt.org/) (Scala Build Tool).

## Building

To compile the project and create a runnable JAR file, use the `sbt assembly` command:

```bash
sbt assembly
```

This will generate a fat JAR file in the `target/scala-3.x.x/` directory (e.g., `target/scala-3.5.1/llmtest-assembly-0.1.0.jar`).

## Running

You can run the Llama model using the assembled JAR file. You must provide the path to the model file (`.gguf` format) using the `--model` or `-m` argument.

Make sure to include the `--add-modules=jdk.incubator.vector` JVM option when running.

**Example (Interactive Mode):**

```bash
java --add-modules=jdk.incubator.vector -jar target/scala-3.5.1/llmtest-assembly-0.1.0.jar --model /path/to/your/model.gguf
```

**Example (Single Prompt Mode):**

```bash
java --add-modules=jdk.incubator.vector -jar target/scala-3.5.1/llmtest-assembly-0.1.0.jar \
  --model /path/to/your/model.gguf \
  --prompt "Translate the following English text to French: 'Hello world!'"
```

### Command-Line Options

*   `--model <path>`, `-m <path>`: (Required) Path to the model file in GGUF format.
*   `--prompt <text>`, `-p <text>`: Run in single-prompt mode with the given text. If omitted, runs in interactive mode.
*   `--system-prompt <text>`: Set a system prompt for the model.
*   `--temperature <float>`: Sampling temperature (default: 0.1).
*   `--topp <float>`: Top-P (nucleus) sampling value (default: 0.95).
*   `--seed <long>`: Random seed (default: System.nanoTime).
*   `--max-tokens <int>`: Maximum number of tokens to generate (default: 16384).
*   `--stream <boolean>`: Print tokens as they are generated (default: true).
*   `--echo <boolean>`: Print all tokens (including prompt) to stderr (default: false).

## Community

Join our Discord server to discuss the project, ask questions, and share your results:
[https://discord.com/invite/vgEg2ZtxCw](https://discord.com/invite/vgEg2ZtxCw)

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by and based on the work of:
*   [llama3.java](https://github.com/mukel/llama3.java)
*   [llama2.c](https://github.com/karpathy/llama2.c) by [Andrej Karpathy](https://twitter.com/karpathy) and his [excellent educational videos](https://www.youtube.com/c/AndrejKarpathy).
