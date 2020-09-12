#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <x86intrin.h>

double fact(int x) {
    double res = 1;
    for (int i = 2; i <= x; ++i) {
        res *= i;
    }
    return res;
}

double sinus(double x) {
    double res = 0;
    double sign = 1;
    for (int n = 0; n < 14; ++n) {
        res += sign * pow(x, 2 * n + 1) / fact((2 * n + 1));
        sign *= -1;
    }
    return res;
}

double clock_timer(int count) {
    clock_t start = clock();
    while (count--) {
        sinus(3);
    }
    clock_t stop = clock();
    return (double) (stop - start) / CLOCKS_PER_SEC;
}

double get_time_of_day_timer(int count) {
    struct timeval tv1, tv2;
    struct timezone tz;
    gettimeofday(&tv1, &tz);
    while (count--) {
        sinus(3);
    }
    gettimeofday(&tv2, &tz);
    return (double)(tv2.tv_sec - tv1.tv_sec) + (double)(tv2.tv_usec - tv1.tv_usec) / 1000000;
}

double TSC_timer(int count) {
    __uint64_t start = __rdtsc();
    while (count--) {
        sinus(3);
    }
    return (double) (__rdtsc() - start) / 2800000000; // 2.8 Ghz CPU
}

int main() {
//    for (int i = 1; i < 10; ++i) {
//        int iters = i;
//        printf("%d cycles:\n", iters);
//        printf("%.8f\n", clock_timer(iters));
//        printf("%.8f\n", get_time_of_day_timer(iters));
//        printf("%.8f\n\n", TSC_timer(iters));
//    }
//    int f1 = 1, f2 = 2;
//    int iters = 38;
//    for (int i = 0; i < iters; ++i) {
//        printf("%d\n", f1);
//        int tmp = f1;
//        f1 = f2;
//        f2 += tmp;
//    }
//    f1 = 1;
//    f2 = 2;
//    printf("===============\n");
//    for (int i = 0; i < iters; ++i) {
//        printf("%.8f\n", clock_timer(f1));
//        int tmp = f1;
//        f1 = f2;
//        f2 += tmp;
//    }
//    f1 = 1;
//    f2 = 2;
//    printf("===============\n");
//    for (int i = 0; i < iters; ++i) {
//        printf("%.8f\n", get_time_of_day_timer(f1));
//        int tmp = f1;
//        f1 = f2;
//        f2 += tmp;
//    }
//    f1 = 1;
//    f2 = 2;
//    printf("===============\n");
//    for (int i = 0; i < iters; ++i) {
//        printf("%.8f\n", TSC_timer(f1));
//        int tmp = f1;
//        f1 = f2;
//        f2 += tmp;
//    }
//    f1 = 1;
//    f2 = 2;
//    printf("===============\n");
    int count = 1000;
    for (int i = 0; i < 4; ++i) {
        printf("Clock timer: %.8f\n"
               "Get_time_of_day timer: %.8f\n"
               "TSC timer: %.8f\n\n",
               clock_timer(count), get_time_of_day_timer(count), TSC_timer(count));
    }
}