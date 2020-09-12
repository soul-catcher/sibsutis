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
    out << (n % 2 != 0 ? 0 : n / 2);

}
