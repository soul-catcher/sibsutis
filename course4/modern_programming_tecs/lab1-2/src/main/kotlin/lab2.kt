import kotlin.math.max
import kotlin.math.min

fun maximum(a: Int, b: Int): Int {
    return max(a, b)
}

fun sumOfSomeTwoDimArrElements(twoDimArr: Array<DoubleArray>): Double {
    val maxLineInd = twoDimArr.maxOf { it.size } - 1
    var sum = 0.0
    for ((i, line) in twoDimArr.withIndex()) {
        for ((j, elem) in line.withIndex()) {
            if (i + j == maxLineInd) {
                sum += elem
            }
        }
    }
    return sum
}

fun minOfAllUpperOfSideDiagonalOfMatrix(matrix: Array<DoubleArray>): Double {
    var min = Double.MAX_VALUE
    for ((i, line) in matrix.withIndex()) {
        min = min(min, line.copyOf(matrix.size - i).minOrNull()!!)
    }
    return min
}
