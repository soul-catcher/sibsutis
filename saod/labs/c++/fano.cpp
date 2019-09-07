#include <math.h>
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>

using namespace std;

int C[50][10] = {{0}};
int Length[50], i;
float SL, SR;

float chas[50] = {0};
char ch[50];


int Med(int L, int R) {
    int k, m1;
    SL = 0;
    for (k = L; k <= R - 1; k++) SL = SL + chas[k];
    if (SL == 0) return 0;
    SR = chas[R];
    m1 = R;
    while (SL >= SR) {
        m1 = m1 - 1;
        SL = SL - chas[m1];
        SR = SR + chas[m1];
    }
    return m1;
}

void Fano(int L, int R, int k) {
    int m;
    if (L < R) {
        k++;
        m = Med(L, R);
        for (i = L; i <= m; i++) {
            C[i][k] = 0;
            Length[i] = Length[i] + 1;
        }
        for (i = m + 1; i <= R; i++) {
            C[i][k] = 1;
            Length[i] = Length[i] + 1;
        }
        Fano(L, m, k);
        Fano(m + 1, R, k);
    }
}

void BubbleSort2(float *A, int n) {
    int i, j;
    float t;
    char t1;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - 1; j++)
            if (A[j + 1] > A[j]) {
                t = A[j + 1];
                A[j + 1] = A[j];
                A[j] = t;
                t1 = ch[j + 1];
                ch[j + 1] = ch[j];
                ch[j] = t1;
            }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    float H = 0.0, Lsr = 0.0;
    int j = 0, q = 0, kol = 0, sum = 0;
    ifstream f("1.txt", ios::out | ios::binary);
    map<char, int> m;
    while (!f.eof()) {
        char c = f.get();
        if ((c == 10) || (c == 13)) continue;
        m[c]++;
        kol++;
    }
    for (map<char, int>::iterator itr = m.begin(); itr != m.end(); j++, ++itr) {
        ch[j] = itr->first;
        chas[j] = (float) itr->second / kol;
    }
    BubbleSort2(chas, 50);
    Fano(0, 49, -1);
    cout << "Символ " << "Верояность символа  " << "Код символа" << endl;
    for (i = 0; i < 50; i++) {
        cout << "    " << ch[i] << "     ";
        printf("%.8f", chas[i]);
        cout << "       ";
        for (j = 0; j < Length[i]; j++) {
            if ((i > 10) && (j == 0)) cout << "1";
            else cout << C[i][j];
        }
        cout << endl;
    }
    for (i = 0; i < m.size(); i++) H = H - chas[i] * (log(chas[i]) / log(2.0));
    for (i = 0; i < m.size(); i++) Lsr = Lsr + chas[i] * Length[i];
    cout << "Kol-vo = " << kol << endl;
    cout << "Size = " << m.size() << endl;
    cout << "Энтропия исходного файла ровна " << H << endl;
    cout << "Средняя длина кодового слова ровна " << Lsr << endl;
    cout << "Соотношение " << Lsr << "<" << H << "+ 1" << " выполнено" << endl;
}
