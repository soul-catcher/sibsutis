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
    int x1, x2, y1, y2, z1, z2;
    in >> x1 >> x2 >> y1 >> y2 >> z1 >> z2;
    double a, b, c, p, s;
    a = sqrt(pow(x2 - y2, 2) + pow(x1 - y1, 2));
    b = sqrt(pow(y2 - z2, 2) + pow(y1 - z1, 2));
    c = sqrt(pow(x2 - z2, 2) + pow(x1 - z1, 2));

    p = (a + b + c) / 2;
    s = sqrt(p *  (p - a) * (p - b) * (p - c));
    out << int(round(s * 4));

}