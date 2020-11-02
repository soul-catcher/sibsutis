#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void arrayRandomFill(double array[], int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = (double) rand() / RAND_MAX;
    }
}

void matrixRandomFill(double *matrix[], int order) {
    for (int i = 0; i < order; ++i) {
        arrayRandomFill(matrix[i], order);
    }
}

// Самое обычное перемножение матриц, имеет наихудшую производительность
void dgemm1(double *matrix1[], double *matrix2[], double *resultMatrix[], int order) {
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            for (int k = 0; k < order; k++) {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

// Улучшенное перемножение
void dgemm2(double *matrix1[], double *matrix2[], double *resultMatrix[], int order) {
    for (int i = 0; i < order; i++) {
        for (int k = 0; k < order; k++) {      // По сравнению с предыдущим вариантом,
            for (int j = 0; j < order; j++) {  // тут поменялись местами эти 2 строки
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

// Блочный алгоритм умножения. Самый быстрый из всех.
void dgemm3(double matrix1[], double matrix2[], double resultMatrix[], int order, int blockSize) {
    int i0, k0, j0;
    double *a0, *c0, *b0;
    for (int i = 0; i < order; i += blockSize) {
        for (int j = 0; j < order; j += blockSize) {
            for (int k = 0; k < order; k += blockSize) {
                for (i0 = 0, c0 = (resultMatrix + i * order + j), a0 = (matrix1 + i * order + k);
                     i0 < blockSize; ++i0, c0 += order, a0 += order) {
                    for (k0 = 0, b0 = (matrix2 + k * order + j); k0 < blockSize; ++k0, b0 += order) {
                        for (j0 = 0; j0 < blockSize; ++j0) {
                            c0[j0] += a0[k0] * b0[j0];
                        }
                    }
                }
            }
        }
    }
}

// Запускает dgemm1 или dgemm2 и возвращает время выполнения умножения
// Параметры: order - порядок матрицы, параметр dgemm определяет какой алгоритм запустить 1 или 2
double simpleMultiplicationTimer(int order, int dgemm) {
    // Выделение динамической памяти, так как большие матрицы не влезают в стек
    double *matrix1[order];
    double *matrix2[order];
    double *resultMatrix[order];
    for (int i = 0; i < order; ++i) {
        matrix1[i] = malloc(order * sizeof(double));
        matrix2[i] = malloc(order * sizeof(double));
        // Матрицу с результатом умножения надо инициализировать нулями, т.е. при помощи calloc, а не malloc,
        // так как каждый элемент матрицы умножения является суммой произведений
        resultMatrix[i] = calloc(order, sizeof(double));
    }

    matrixRandomFill(matrix1, order);
    matrixRandomFill(matrix2, order);

    clock_t start = clock();
    if (dgemm == 1) {  // Выбор алгоритма в зависимости от параметра
        dgemm1(matrix1, matrix2, resultMatrix, order);
    } else if (dgemm == 2) {
        dgemm2(matrix1, matrix2, resultMatrix, order);
    }
    clock_t stop = clock();

    for (int i = 0; i < order; ++i) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(resultMatrix[i]);
    }
    return (double) (stop - start) / CLOCKS_PER_SEC;  // Время выполнения (в секундах)
}

// Запускает dgemm3 и возвращает время blockSize должен быть кратен порядку матрицы
double blockMultiplicationTimer(int order, int blockSize) {
    // В отличие от предыдущих алгоритмов, этот требует представление матриц в виде одномерных массивов размером
    // квадрата порядка матрицы.
    int size = order * order;

    // Выделение динамической памяти, так как большие матрицы не влезут в стек
    double *matrix1 = malloc(size * sizeof(double));
    double *matrix2 = malloc(size * sizeof(double));
    // Матрицу с результатом умножения надо инициализировать нулями, т.е. при помощи calloc, а не malloc
    double *resultMatrix = calloc(size, sizeof(double));

    clock_t start = clock();
    dgemm3(matrix1, matrix2, resultMatrix, order, blockSize);
    clock_t stop = clock();

    free(matrix1);
    free(matrix2);
    free(resultMatrix);

    return (double) (stop - start) / CLOCKS_PER_SEC;
}

// Часть 1 - ввод порядка матриц и размера блока с клавиатуры
// Выводит на экран время работы трёх способов умножения
void part1() {
    int order, blockSize;
    printf("Input matrix order: ");
    scanf("%d", &order);
    printf("Input size of block: ");
    scanf("%d", &blockSize);

    double simpleTime = simpleMultiplicationTimer(order, 1);
    double strByStrTime = simpleMultiplicationTimer(order, 2);
    double blockTime = blockMultiplicationTimer(order, blockSize);
    printf("\nSimple (dgemm 1) multiplication time: %f sec\n", simpleTime);
    printf("String by string multiplication time: %f sec\n", strByStrTime);
    printf("Block multiplication time: %f sec\n", blockTime);
}

void part2() {
    int order = 2048;

    printf("dgemm1\n");
    printf("%f sec\n", simpleMultiplicationTimer(order, 1));

    printf("dgemm2\n");
    printf("%f sec\n", simpleMultiplicationTimer(order, 2));

    printf("dgemm3\n");
    for (int blockSize = 1; blockSize <= order; blockSize *= 2) {
        printf("%f sec\n", blockMultiplicationTimer(order, blockSize));
    }
}

int main() {
    part2();
}