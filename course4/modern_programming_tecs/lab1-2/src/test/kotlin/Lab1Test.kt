import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class Lab1Test {

    @Test
    fun arraySum() {
        assertArrayEquals(intArrayOf(8, 10, 12), arraySum(intArrayOf(3, 4, 5), intArrayOf(5, 6, 7)))
    }

    @Test
    fun arrayShiftLeft() {
        val arr = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        arrayShiftLeft(arr, 2)
        assertArrayEquals(doubleArrayOf(3.0, 4.0, 5.0, 1.0, 2.0), arr)
    }

    @Test
    fun indexOfSubArray() {
        assertEquals(3, indexOfSubArray(intArrayOf(1, 2), intArrayOf(5, 5, 3, 1, 2, 5, 5)))
    }
}
