#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

int system_closed = 0;// 0 - закрыта, 1 - фиктивный поставщик, 2 - фиктивный потребитель

void read_from_file(vector<vector<int>> &C, vector<int> &A, vector<int> &B, char file[]) {
    ifstream in(file);
    int m, n, i, j;// m - строки, n - столбцы
    in >> m >> n;
    for (i = 0; i < A.size(); i++) {
        in >> A[i];
        //cout<<A[i]<<" ";
    }
    for (i = 0; i < B.size(); i++) {
        in >> B[i];
        //cout<<B[i]<<" ";
    }
    for (i = 0; i < C.size(); i++) {
        for (j = 0; j < C[0].size(); j++) {
            in >> C[i][j];
            //cout<<setw(2)<<C[i][j]<<" ";
        }//cout<<endl;
    }
}

void print_table(vector<vector<int>> &C, vector<vector<int>> &X, vector<int> &A, vector<int> &B) {
    cout << endl
         << setw(11) << " " << setw(1) << "|" << setw(static_cast<int>(B.size()) * 6) << "Потребители" << setw(1)
         << "|" << setw(6) << " " << endl
         << setw(21) << "Поставщики" << setw(1) << "|";
    for (int i = 0; i < (B.size()) * 4 - 1; i++) cout << setw(1) << "-";
    cout << setw(1) << "|" << setw(6) << "Запасы" << endl
         << setw(11) << " " << setw(1) << "|";
    for (int i = 0; i < B.size(); i++) cout << setw(2) << "B" << setw(1) << i + 1 << "|";
    cout << setw(6) << " " << endl;
    for (int i = 0; i < (B.size()) * 4 + 11 + 6 + 2; i++) cout << setw(1) << "-";
    cout << endl;
    for (int i = 0; i < A.size(); i++) {
        cout << setw(10) << "A" << setw(1) << i + 1 << setw(1) << "|";
        for (int j = 0; j < B.size(); j++) cout << setw(3) << C[i][j] << "|";
        cout << setw(6) << A[i] << endl;
        cout << setw(11) << " " << setw(1) << "|";
        for (int j = 0; j < B.size(); j++) {
            if (X[i][j] == -1) cout << setw(1) << "-" << setw(3) << "|";
            else if (X[i][j] == -2) cout << setw(3) << " " << setw(1) << "|";
            else cout << setw(3) << X[i][j] << "|";
        }
        cout << endl;
        for (int k = 0; k < B.size() * 4 + 11 + 6 + 2; k++) cout << setw(1) << "-";
        cout << endl;
    }
    cout << setw(11) << "Потребности" << setw(1) << "|";
    for (int i : B) cout << setw(3) << i << "|";
    cout << setw(6) << " " << endl << endl;
}

int is_system_closed(vector<int> &A, vector<int> &B) {
    int sumA = 0, sumB = 0;
    for (int i : A) sumA += i;
    for (int i : B) sumB += i;
    cout << "Сумма ai = " << sumA << "; ";
    cout << "сумма bi = " << sumB << endl;
    return sumA - sumB;
}

void addA(vector<vector<int>> &C, vector<vector<int>> &X, vector<int> &A, int a) {
    vector<int> c(C[0].size(), 0);
    vector<int> x(C[0].size(), -2);
    C.push_back(c);
    X.push_back(x);
    A.push_back(a);
}

void addB(vector<vector<int>> &C, vector<vector<int>> &X, vector<int> &B, int b) {
    for (int i = 0; i < C.size(); i++) {
        C[i].push_back(0);
        X[i].push_back(-2);
    }
    B.push_back(b);
}

void remove(vector<vector<int>> &X, int row, int column) {
    if (row != -1) {
        for (int &i : X[row]) {
            if (i == -2) i = -1;
        }
    }
    if (column != -1) {
        for (auto &i : X) {
            if (i[column] == -2) i[column] = -1;
        }
    }
}

void countZ(vector<vector<int>> &C, vector<vector<int>> &X) {
    int Z = 0;
    bool first = true;
    cout << "Z = ";
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            if (X[i][j] >= 0) {
                Z += X[i][j] * C[i][j];
                if (first) {
                    cout << C[i][j] << "*" << X[i][j];
                    first = false;
                } else {
                    cout << " + " << C[i][j] << "*" << X[i][j];
                }
            }
        }
    }
    cout << " = " << Z << endl;
}

void minimum_ij(vector<vector<int>> &C, vector<vector<int>> &X, int &k, int &l, int flag) {
    int min = 100000, i1 = 0, j1 = 0, i2 = -1, j2 = -1;
    if (flag == 1) i1++;
    else if (flag == 2) j1++;
    for (int i = 0; i < C.size() - i1; i++) {
        for (int j = 0; j < C[0].size() - j1; j++) {
            if (X[i][j] == -2) {
                if (C[i][j] < min) {
                    min = C[i][j];
                    i2 = i;
                    j2 = j;
                    //cout<< min<<endl;
                    //cout<< i<< " "<<j<<endl;
                }
            }
        }
    }
    if (i2 == -1 && j2 == -1) {
        if (flag == 1) {
            for (int j = 0; j < C[0].size(); j++) {
                if (X[X.size() - 1][j] == -2) {
                    if (C[X.size() - 1][j] < min) {
                        min = C[X.size() - 1][j];
                        i2 = static_cast<int>(X.size()) - 1;
                        j2 = j;
                    }
                }
            }
        } else if (flag == 2) {
            for (int i = 0; i < C.size(); i++) {
                if (X[i][X.size() - 1] == -2) {
                    if (C[i][X.size() - 1] < min) {
                        min = C[i][X.size() - 1];
                        i2 = i;
                        j2 = static_cast<int>(X.size()) - 1;
                    }
                }
            }
        }
    }
    //cout<< i2<<" "<< j2<<endl;
    k = i2;
    l = j2;
}

void minimum_cost(vector<vector<int>> &C, vector<vector<int>> &X, vector<int> &A, vector<int> &B) {
    int i = 0, j = 0, m;
    bool nul = false;
    cout << "Метод минимальной стоимости: " << endl;
    cout << "Должно быть " << A.size() << " + " << B.size() << " - 1 = " << A.size() + B.size() - 1
         << " заполненных клеток" << endl;
    for (int k = 0; k < (A.size() + B.size() - 1); k++) { //m+n-1
        minimum_ij(C, X, i, j, system_closed);
        cout << "Работаем с клеткой C[" << i + 1 << "][" << j + 1 << "] :" << endl;
        m = min(A[i], B[j]);
        cout << "min(A[" << i + 1 << "];B[" << j + 1 << "]) = min(" << A[i] << ";" << B[j] << ") = " << m << endl
             << "A[" << i + 1 << "] = " << A[i] << " - " << m << " = " << A[i] - m << endl
             << "B[" << j + 1 << "] = " << B[j] << " - " << m << " = " << B[j] - m << endl;
        A[i] -= m;
        B[j] -= m;
        X[i][j] = m;
        if (A[i] == 0 && B[j] == 0) {
            k++;
            for (int g = static_cast<int>(C[0].size()) - 1; g >= 0; g--) {
                if (X[i][g] == -2) {
                    X[i][g] = 0;
                    nul = true;
                    remove(X, i, j);
                    break;
                }
            }
            if (!nul) {
                for (int g = static_cast<int>(C.size()) - 1; g >= 0; g--) {
                    if (X[g][j] == -2) {
                        X[g][j] = 0;
                        remove(X, i, j);
                        break;
                    }
                }
            }
            nul = false;
        } else if (A[i] == 0) {
            remove(X, i, -1);
            //i++;
        } else if (B[j] == 0) {
            remove(X, -1, j);
            //j++;
        }
        print_table(C, X, A, B);
    }
}

int main(int argc, char *argv[]) {
    ifstream in(argv[1]);
    int m, n, close;// m - строки, n - столбцы
    in >> m >> n;
    vector<vector<int>> C(m, vector<int>(n));//матрица тарифов
    vector<vector<int>> X(m, vector<int>(n, -2));//матрица перевозок
    vector<int> A(m);//запасы
    vector<int> B(n);//потребности
    cout << "Начальное условие:" << endl;
    read_from_file(C, A, B, argv[1]);
    print_table(C, X, A, B);
    //cout<<is_system_closed(A,B);
    close = is_system_closed(A, B);
    if (close < 0) {
        system_closed = 1;
        addA(C, X, A, close * (-1));
        cout << "Вводим фиктивного поставщика с запасом " << close * (-1) << ":" << endl;
        print_table(C, X, A, B);
    } else if (close > 0) {
        system_closed = 2;
        addB(C, X, B, close);
        cout << "Вводим фиктивного потребителя с потребностью " << close << ":" << endl;
        print_table(C, X, A, B);
    }
    minimum_cost(C, X, A, B);
    countZ(C, X);
}
