package ai.xpress.llama4s.utils

import org.json4s._

extension (json: JValue) {
  def --(key: String): JValue = {
    json match {
      case JObject(l) =>
        JObject(
          l.filter { (name, _) =>
            name != key
          }
        )
      case _ =>
        json
    }
  }
}
