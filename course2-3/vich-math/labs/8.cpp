#include <iostream>
#include <fstream>
#include <cmath>
typedef unsigned long long int ull;
using namespace std;

struct point {
    int x;
    int y;
};

double angle_point (point a, point b, point c)
{
    double x1 = a.x - b.x, x2 = c.x - b.x;
    double y1 = a.y - b.y, y2 = c.y - b.y;
    double d1 = sqrt (x1 * x1 + y1 * y1);
    double d2 = sqrt (x2 * x2 + y2 * y2);
    double j = (x1 * x2 + y1 * y2) / (d1 * d2);
    return acos(j);
}
int main() {
    ifstream in;
    ofstream out;
    in.open("input.txt");
    out.open("output.txt");

    int n;
    in >> n;
    point points[n];
    for (auto &p : points) {
        in >> p.x;
        in >> p.y;
    }

    int l = 0, r = 0;
    for (int i = 2; i < n; ++i) {
        cout << angle_point(points[i - 2], points[i - 1], points[i])<< ' ';
    }

}