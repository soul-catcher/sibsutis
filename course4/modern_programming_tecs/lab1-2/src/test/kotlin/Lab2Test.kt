import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

internal class Lab2Test {

    @Test
    fun maximum() {
        assertEquals(8, maximum(3, 8))
    }

    @Test
    fun sumOfSomeTwoDimArrElements() {
        val twoDimArr = arrayOf(
            doubleArrayOf(3.1, 3.0),
            doubleArrayOf(3.0, 5.3, 1.1, 4.6),
            doubleArrayOf(2.2, 4.4)
        )
        assertEquals(1.1 + 4.4, sumOfSomeTwoDimArrElements(twoDimArr))
    }

    @Test
    fun minOfAllUpperOfSideDiagonalOfMatrix() {
        val matrix = arrayOf(
            doubleArrayOf(3.1, 3.0, 6.6),
            doubleArrayOf(3.0, 5.3, 1.1),
            doubleArrayOf(2.2, 4.4, 4.6)
        )
        assertEquals(2.2, minOfAllUpperOfSideDiagonalOfMatrix(matrix))
    }
}
