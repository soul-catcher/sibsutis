import java.util.Collections

fun arraySum(arr1: IntArray, arr2: IntArray): IntArray {
    return arr1.zip(arr2, Int::plus).toIntArray()
}

fun arrayShiftLeft(arr: DoubleArray, n: Int) {
    val list = arr.toList()
    Collections.rotate(list, -n)
    list.toDoubleArray().copyInto(arr)
}

fun indexOfSubArray(seq: IntArray, vec: IntArray): Int {
    return Collections.indexOfSubList(vec.toList(), seq.toList())
}
