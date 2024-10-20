package ai.xpress.llama4s.utils

extension [A, B](tmap: Map[A, B]) {
  def inverse: Map[B, A] = {
    tmap.map { (k, v) =>
      (v, k)
    }
  }
}

extension [A](arr: Array[A]) {
  def swap(i: Int, j: Int): Unit = {
    val tmp = arr(i)
    arr(i) = arr(j)
    arr(j) = tmp
  }

  def siftDown(from: Int, n: Int)(using ord: Ordering[A]): Unit = {
    var prev = from
    var next = 0
    while ({
      next = 2 * prev + 1;
      next
    } < n) {
      var r = 2 * prev + 2
      if (r < n && ord.compare(arr(r), arr(next)) < 0) {
        next = r;
      }
      if (ord.compare(arr(next), arr(prev)) < 0) {
        arr.swap(prev, next)
        prev = next;
      } else {
        return
      }
    }
  }
}

extension [A](ordT: Ordering.type) {
  def map[B](conv: A => B)(using ord: Ordering[B]): Ordering[A] = {
    new Ordering[A] {
      def compare(x: A, y: A): Int = {
        ord.compare(conv(x), conv(y))
      }
    }
  }
}
