#include <iostream>
#include <vector>
#include <cmath>
#include <x86intrin.h>
#include <iomanip>

using namespace std;

void saxpy_blas(const vector<int> &x, vector<int> &y, const int a) {
    for (unsigned long i = 0; i < y.size(); ++i) {
        y[i] = a * x[i] + y[i];
    }
}

vector<vector<double>> dgemm_blas(const vector<vector<double>> &x, const vector<vector<double>> &y) {
    vector<vector<double>> ans(x.size(), vector<double>(y[0].size(), 0));
    for (unsigned long row = 0; row < x.size(); ++row) {
        for (unsigned long column = 0; column < y[0].size(); ++column) {
            for (unsigned long i = 0; i < y.size(); ++i) {
                ans[row][column] += x[row][i] * y[i][column];
            }
        }
    }
    return ans;
}

double TSC_timer_vec(const vector<int> &x, vector<int> &y, const int a) {
    __uint64_t start = __rdtsc();
    saxpy_blas(x, y, a);
    return (__rdtsc() - start) / 2800000000.; // 2.8 Ghz CPU
}

double TSC_timer_matr(const vector<vector<double>> &x, const vector<vector<double>> &y) {
    __uint64_t start = __rdtsc();
    dgemm_blas(x, y);
    return (__rdtsc() - start) / 2800000000.; // 2.8 Ghz CPU
}

void p1() {
    cout << "Input size of vectors: ";
    unsigned long size;
    cin >> size;
    vector<int> x(size), y(size);
    for (unsigned long i = 0; i < size; ++i) {
        x[i] = rand() % 10;
        y[i] = rand() % 10;
    }
    cout << "1st vector: ";
    for (auto elem : x) {
        cout << elem << ' ';
    }
    cout << "\n2nd vector: ";
    for (auto elem : y) {
        cout << elem << ' ';
    }
    TSC_timer_vec(x, y, 10);
    cout << "\nresult: ";
    for (auto elem : y) {
        cout << elem << ' ';
    }
}

void p2() {
    for (int i = 2; i < 104; ++i) {
        unsigned long size = round(exp(i / 4.983));
        vector<int> x(size), y(size);
//        cout << size << '\n';
        for (unsigned long j = 0; j < size; ++j) {
            x[j] = rand() % 1000;
            y[j] = rand() % 1000;
        }
        cout << fixed << setprecision(8) << TSC_timer_vec(x, y, 10) << '\n';
    }
}

void p3() {
    cout << "Input size of first matrix, then size of second.\n"
            "Number of columns of first matrix and rows of second matrix should be equal.\n"
            "Numbers should be separated with any whitespase characters.\n";
    unsigned long rows1, col1, rows2, col2;
    cin >> rows1 >> col1 >> rows2 >> col2;
    vector<vector<double>> x(rows1, vector<double>(col1)), y(rows2, vector<double>(col2));
    cout << "First matr:\n";
    for (auto &row : x) {
        for (auto &elem : row) {
            elem = static_cast<double>(rand()) / RAND_MAX;
            cout << elem << ' ';
        }
        cout << '\n';
    }
    cout << "Second matr:\n";
    for (auto &row : y) {
        for (auto &elem : row) {
            elem = static_cast<double>(rand()) / RAND_MAX;
            cout << elem << ' ';
        }
        cout << '\n';
    }
    auto ans = dgemm_blas(x, y);
    cout << "Result:\n";
    for (auto &row : ans) {
        for (auto &elem : row) {
            cout << elem << ' ';
        }
        cout << '\n';
    }
}

void p4() {
    for (int i = 0; i < 133; ++i) {
        unsigned long size = round(exp(i / 18.0495039139));
        vector<vector<double>> x(size, vector<double>(size)), y(size, vector<double>(size));
//        cout << size << '\n';
        for (unsigned long j = 0; j < size; ++j) {
            for (unsigned long k = 0; k < size; ++k) {
                x[j][k] = static_cast<double>(rand()) / RAND_MAX;
                y[j][k] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
        cout << fixed << setprecision(8) << TSC_timer_matr(x, y) << '\n';
    }
}

int main() {
    p4();
}

