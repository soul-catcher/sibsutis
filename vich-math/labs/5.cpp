#include <iostream>
#include <fstream>
#include <cmath>
typedef unsigned long long int ull;
typedef long long ll;
using namespace std;

bool prime(ll n){
    for(ll i=2;i<=sqrt(n);i++)
        if(n%i==0)
            return false;
    return true;
}

int main() {
    ifstream in;
    ofstream out;
    in.open("input.txt");
    out.open("output.txt");
    int n;
    in >> n;
    if (n == 1) {
        out << "NO";
        return 0;
    }
    string str = to_string(n);
    bool fl = true;
    int len = str.length();
    for (int i = 0; i < len / 2 + 1 and fl; i++) {
        fl = fl && str[i] == str[len - i - 1];
    }
    out << (fl && prime(n) ? "YES" : "NO");
}