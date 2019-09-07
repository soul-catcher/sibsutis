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
    string str;
    in >> str;
    int ans = 0;
    for (auto let : str) {
        ans += let - 0x60;
    }
    out << ans;
}