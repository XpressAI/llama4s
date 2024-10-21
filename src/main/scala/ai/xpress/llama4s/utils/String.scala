package ai.xpress.llama4s.utils

import scala.collection.mutable.{Buffer => MBuf}
import java.lang.{StringBuilder => JStringBuilder}
import java.util.HexFormat
import java.util.regex.Pattern

extension (str: String) {
  def replaceControlChars: String = {
    // we don't want to print control characters
    // which distort the output (e.g. \n or much worse)
    // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    // http://www.unicode.org/reports/tr44/#GC_Values_Table\
    val codepoints = str.codePoints.toArray
    val chars = new JStringBuilder
    for (cp <- codepoints) {
      if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
        chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)) // escape
      } else {
        chars.appendCodePoint(cp) // this character is ok
      }
    }
    chars.toString
  }
}

extension (pattern: Pattern) {
  def findAll(text: String): Seq[String] = {
    val matches = MBuf.empty[String]
    val matcher = pattern.matcher(text)
    while (matcher.find) {
      matches += matcher.group
    }
    matches.toSeq
  }
}
