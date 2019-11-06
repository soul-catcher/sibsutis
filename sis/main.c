#include <stdio.h>

void comb_sort(int arr[], int n) {
    int tmp, k;
    int step = n;
    while (n > 1) {
        step /= 1.247f;
        if (step < 1) {
            step = 1;
        }
        k = 0;
        for (int i = 0; i + step < n; ++i) {
            if (arr[i] > arr[i + step]) {
                tmp = arr[i];
                arr[i] = arr[i + step];
                arr[i + step] = tmp;
                k = i;
            }
        }
        if (step == 1)
            n = k + 1;
    }
}

int main() {
    const int size = 6;
    int arr[] = {1, 3, 7, 2, 4, 6};
    comb_sort(arr, size);
    for (int i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
}

