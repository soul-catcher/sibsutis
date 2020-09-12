#include <iostream>
#include <fstream>
#include <cmath>
typedef unsigned long long int ull;
using namespace std;

ull fib(ull n) {
    //F(n)
    ull x = 1;
    //F(n-1)
    ull y = 0;
    for (int i = 0; i < n; i++)
    {
        x += y;
        y = x - y;
    }
    return y;
}


int main() {
    ifstream in;
    ofstream out;
    in.open("input.txt");
    out.open("output.txt");
    ull st, fn;
    in >> st >> fn;

    ull left = 1;
    ull right = ull(pow(10, 8));
    ull mid;
    while (left < right) {
        mid = (left + right) / 2;
        ull f_mid = fib(mid);
        if (f_mid == st) {
            break;
        }
        if (f_mid < st) {
            left = mid + 1;
        } else if (f_mid > st)
            right = mid;
    }
    ull ans = 0;
    while (fib(mid) <= fn) {
        ans += fib(mid++);
    }

    out << ans;
}