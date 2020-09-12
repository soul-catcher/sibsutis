#include <iostream>

enum Order {
    from_center, from_left_up_corner
};

void helix(int **matrix, int size, int *array, Order order) {
    int y = -1, x = 0, d = 1, counter, modifier;
    switch (order) {
        case from_left_up_corner:
            counter = -1;
            modifier = 1;
            break;
        case from_center:
            counter = size * size;
            modifier = -1;
    }

    for (int i = 0; i < size;) {
        for (int j = i; j < size; j++) {
            y += d;
            array[counter += modifier] = matrix[x][y];
        }
        i++;
        for (int j = i; j < size; j++) {
            x += d;
            array[counter += modifier] = matrix[x][y];
        }
        d *= -1;
    }
}

void first(int size) {
    int N = size * size * 4;
    auto array2d = new int *[size];
    for (int i = 0; i < size; i++) {
        array2d[i] = new int[size];
        for (int j = 0; j < size; j++) {
            array2d[i][j] = rand() % 90 + 10;
        }
    }

    auto array = new int[N];
    int iter = 0;

    for (int x = size - 1; x >= 0; x--) {
        int xLoc = x;
        for (int y = 0; y < size - x; ++y) {
            array[iter++] = array2d[y][xLoc++];
        }
    }
    for (int y = 1; y < size; y++) {
        int yLoc = y;
        for (int x = 0; x < size - y; ++x) {
            array[iter++] = array2d[yLoc++][x];
        }
    }

    for (int x = 0; x < size; x++) {
        int xLoc = x;
        for (int y = 0; y < x + 1; y++) {
            array[iter++] = array2d[y][xLoc--];
        }
    }
    for (int y = 1; y < size; y++) {
        int yLoc = y;
        for (int x = size - 1; x >= y; x--) {
            array[iter++] = array2d[yLoc++][x];
        }
    }

    helix(array2d, size, array + iter, from_center);
    iter += size * size;
    helix(array2d, size, array + iter, from_left_up_corner);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << array2d[i][j] << ' ';
        }
        delete[] array2d[i];
        std::cout << '\n';
    }
    delete[] array2d;
    std::cout << '\n';
    for (int i = 0; i < 4; i++) {
        for (int j = size * size * i; j < size * size * (i + 1); ++j) {
            std::cout << array[j] << ' ';
        }
        std::cout << '\n';
    }
    delete[] array;
    std::cout << '\n';
}

void second() {
    int nStr;
    std::cout << "Number of strings: ";
    std::cin >> nStr;
    auto array2d = new int *[nStr];

    for (int i = 0; i < nStr; i++) {
        int nElem;
        std::cout << "Number of elements in " << i + 1 << " string: ";
        std::cin >> nElem;
        array2d[i] = new int[nElem + 1];
        array2d[i][0] = nElem;
        for (int j = 1; j <= nElem; j++) {
            int element;
            std::cout << "Input " << j << " element: ";
            std::cin >> element;
            array2d[i][j] = element;
        }
    }

    for (int i = 0; i < nStr; i++) {
        for (int j = 1; j <= array2d[i][0]; j++) {
            std::cout << array2d[i][j] << ' ';
        }
        delete[] array2d[i];
        std::cout << '\n';
    }
    delete[] array2d;
}

int main() {
    first(4);
    second();
}