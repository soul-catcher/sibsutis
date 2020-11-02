#include <iostream>
#include <fstream>
#include <cmath>
typedef unsigned long long int ull;
using namespace std;

ull factorial(int i)
{
    if (i==0) return 1;
    else return i*factorial(i-1);
}
int main() {
    ifstream in;
    ofstream out;
    in.open("input.txt");
    out.open("output.txt");
    int n;
    in >> n;
    out << (n - 1) * 2 * factorial((n - 2));
}
