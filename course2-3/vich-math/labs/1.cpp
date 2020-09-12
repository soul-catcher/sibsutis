#include <iostream>
#include <fstream>
#include <cmath>
typedef unsigned long long int ull;
using namespace std;
int main() {
    ifstream in;
    ofstream out;
    in.open("input.txt");
    out.open("output.txt");
    int n;
    in >> n;
    string in7;
    ull ans = 0;
    while (n) {
        in7.append(to_string(n % 7));
        n /= 7;
    }
    for (ull i = in7.length(); i > 0; --i){
        ans += ull((in7[i - 1] - 0x30) * pow(7, in7.length() - i));
    }
    out << ans;
}
