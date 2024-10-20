package ai.xpress.llama4s.model
import scala.math.Pi
import scala.math.cos
import scala.math.pow
import scala.math.sin

object RoPE {
  def precomputeFreqsCis(
      contextLength: Int,
      headSize: Int,
      theta: Double,
      ropeScaling: Boolean,
      scaleFactor: Float,
      loFreqFactor: Float,
      hiFreqFactor: Float,
      oldContextLength: Float
  ): (Array[Float], Array[Float]) = {
    require(headSize % 2 == 0)
    val cr = new Array[Float](contextLength * (headSize / 2))
    val ci = new Array[Float](contextLength * (headSize / 2))
    var n = 0

    for (pos <- 0 until contextLength) {
      for (i <- 0 until headSize by 2) {
        var freq = (1.0f / pow(theta, i / headSize.toDouble)).toFloat
        if (ropeScaling) {
          // Llama 3.1 scaling
          val loFreqWavelen = oldContextLength / loFreqFactor
          val hiFreqWavelen = oldContextLength / hiFreqFactor
          val wavelen = 2.0f * Pi.toFloat / freq
          freq = {
            if (wavelen < hiFreqWavelen) {
              freq
            } else if (wavelen > loFreqWavelen) {
              freq / scaleFactor
            } else {
              val smooth = {
                (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor)
              }
              (1.0f - smooth) * freq / scaleFactor + smooth * freq
            }
          }
        }
        val valFreq = pos * freq
        cr(n) = cos(valFreq).toFloat
        ci(n) = sin(valFreq).toFloat
        n += 1
      }
    }
    require(contextLength * (headSize / 2) == n)
    (cr, ci)
  }
}
