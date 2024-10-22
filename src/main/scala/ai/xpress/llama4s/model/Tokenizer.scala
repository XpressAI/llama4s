package ai.xpress.llama4s.model

import ai.xpress.llama4s.gguf.GGUFFile
import ai.xpress.llama4s.utils._
import scala.collection.mutable.{Buffer => MBuf}
import scala.util.boundary
import scala.util.matching.Regex
import java.lang.{Byte => JByte}
import java.lang.{StringBuilder => JStringBuilder}
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern
import boundary.break

object Tokenizer {
  val Llama3Model = "gpt2"

  val Llama3Pattern = {
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
      .r
  }

  /** Returns list of utf-8 byte and a corresponding list of unicode strings. The reversible bpe codes work on unicode
    * strings. This means you need a large # of unicode characters in your vocab if you want to avoid UNKs. When you're
    * at something like a 10B token dataset you end up needing around 5K for decent coverage. This is a significant
    * percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
    * unicode strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
    */
  val BytesToUnicode: Map[Int, Int] = {
    val bs = ('!'.to('~') ++ '¡'.to('¬') ++ '®'.to('ÿ')).map(_.asInstanceOf[Int])
    val ds = 0.until(256).filterNot(bs.contains(_))

    (bs ++ ds).zip(bs ++ 256.to(256 + ds.size)).toMap
  }

  val ByteEncoder = BytesToUnicode
  val ByteDecoder = BytesToUnicode.inverse

  def fromGGUF(file: GGUFFile, vocabulary: Vocabulary): Tokenizer = {
    val merges = {
      file
        .header
        .metadata
        .apply("tokenizer.ggml.merges")
        .asInstanceOf[Array[String]]
        .map(x => {
          x.split(" ")
        })
        .map { parts =>
          val i1 = vocabulary.indexOf(parts(0)).get
          val i2 = vocabulary.indexOf(parts(1)).get
          // Merge index
          val i3 = vocabulary.indexOf(vocabulary.get(i1) + vocabulary.get(i2)).get
          ((i1, i2), i3)
        }
        .toMap
    }

    val baseTokens = 128000 // Assume all tokens after the base ones are special.
    val specialTokens = {
      vocabulary
        .tokens
        .drop(baseTokens)
        .zipWithIndex
        .map { (tok, i) =>
          (tok, i + baseTokens)
        }
        .toMap
    }

    Tokenizer(vocabulary, merges, Llama3Pattern, specialTokens)
  }

  def merge(ids: Seq[Int], pair: (Int, Int), idx: Int): Seq[Int] = {
    val newids = MBuf.empty[Int]

    var i = 0
    while (i < ids.size) {
      // if not at the very last position AND the pair matches, replace it
      if (i < ids.size - 1 && (ids(i), ids(i + 1)) == pair) {
        newids += idx
        i += 2
      } else {
        newids += ids(i)
        i += 1
      }
    }

    newids.toSeq
  }
}

final case class Tokenizer(
    vocabulary: Vocabulary,
    merges: Map[(Int, Int), Int],
    regex: Regex,
    specialTokens: Map[String, Int]
) {
  val pattern = regex.pattern

  override def toString: String = {
    s"${getClass.getSimpleName}(vocabsize = ${vocabulary.size}, merges = ${merges.size}, specialtokens = ${specialTokens.size})"
  }

  def isSpecialToken(tokenIdx: Int): Boolean = {
    specialTokens.values.toSeq.contains(tokenIdx)
  }

  def encodeChunk(chunk: String): Seq[Int] = {
    // Return the token ids
    // let's begin. first, convert all bytes to integers in range 0..255
    var ids = chunk
      .toCharArray
      .toSeq
      .map { c =>
        vocabulary.indexOf(String.valueOf(c.asInstanceOf[Char])).get
      }

    boundary {
      while (ids.size >= 2) {
        // find the pair with the lowest merge index
        val pair = ids
          .sliding(2)
          .map { w =>
            (w(0), w(1))
          }
          .toSet
          .min(using Ordering.map(merges.getOrElse(_, Int.MaxValue)))

        // subtle: if there are no more merges available, the key will
        // result in an inf for every single pair, and the min will be
        // just the first pair in the list, arbitrarily
        // we can detect this terminating case by a membership check
        if (!merges.contains(pair)) {
          break() // nothing else can be merged anymore
        }

        // Otherwise let's merge the best pair (lowest merge index)
        ids = Tokenizer.merge(ids, pair, merges(pair))
      }
    }

    ids
  }

  def encode(text: String, special: Set[String] = Set.empty): Seq[Int] = {
    assert(special.subsetOf(specialTokens.keySet))

    val cleaned = {
      text
        .getBytes(StandardCharsets.UTF_8)
        .foldLeft(new JStringBuilder) { case (accum, b) =>
          accum.appendCodePoint(Tokenizer.ByteEncoder(JByte.toUnsignedInt(b)))
        }
        .toString
    }

    // Split text into chunks of text by categories defined in regex pattern
    // All chunks of text are encoded separately, then results are joined

    if (special.isEmpty) {
      pattern.findAll(cleaned).flatMap(encodeChunk(_))

    } else {
      val pattern2 = special.map(Pattern.quote(_)).mkString("(", "|", ")")
      cleaned
        .split(pattern2)
        .flatMap { chunk =>
          if (special.contains(chunk)) {
            Seq(specialTokens(chunk))
          } else {
            encodeChunk(chunk)
          }
        }
    }
  }

  def decode(tokens: Seq[Int]): String = {
    val bytes = tokens
      .foldLeft(new StringBuilder) { case (accum, tok) =>
        accum.append(vocabulary.get(tok))
      }
      .toString
      .codePoints
      .map(Tokenizer.ByteDecoder(_))
      .toArray
      .map(_.toByte)

    new String(bytes, StandardCharsets.UTF_8)
  }

  def decode(token: Int): String = {
    decode(Seq(token))
  }
}
