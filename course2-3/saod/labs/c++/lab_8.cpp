#include <iostream>
#include <iomanip>

using namespace std;

void Bin(int x, int n, char *str) {
    int temp;
    for(int i = n-2; i >= 0; i--) {
        temp = (x % 2) + '0';
        str[i] = (char)temp;
        x = x / 2;
        if(x <= 0) {
            for(int j = i-1; j >= 0; j--)
                str[j] = '0';
            break;
        }
    }
}

void fix_var() {
    int n = 16, i, l;
    int index;
    char c[n], e[5], man[n-1];
    for(int x = 0; x < 64; x++) {
        Bin(x, n, c);
        for(i = 0; i < n-1; i++) {
            if(c[i] == '1')
                break;
        }
        index = n - i - 1;
        Bin(index, 5, e);
        printf("%5d|", x);
        cout << " " << e;
        if(x >= 2) {
            l = 0;
            for(int j = i+1; j < n; j++) {
                man[l] = c[j];
                l++;
            }
            cout << " " << man;
        }
        cout << "\n";
    }
    cout << "\n";
}

void gamma(int n) {
    for (int i = 1; i < n; i++) {
        string mant;
        for (int j = i; j >= 1; j /= 2) {
            mant.insert(0, 1, (char)(j % 2 + '0'));
        }
        string exp;
        if (mant.length() >= 1) {
        	exp.insert(0, mant.length() - 1, '0');
        }
        cout << setw(3) << i << setw(6) << exp << " | " <<  mant << '\n';

    }
}

string int_to_str_b(int i) {
    string str;
    for (int j = i; j >= 1; j /= 2) {
        str.insert(0, 1, (char)(j % 2 + '0'));
    }
    return str;
}

void omega(int n) {
    cout << "  1 |  0\n";

    for (int i = 2; i < n; ++i) {
        string str = "0";
        str.insert(0, 1, ' ');


        str.insert(0, int_to_str_b(i));
        string str1 = int_to_str_b(i);
        str.insert(0, 1, ' ');
        int len = (int)str1.length() - 1;
        string tmp=int_to_str_b(len);
        while (len>1){
            str.insert(0, tmp);
            str.insert(0, 1, ' ');
            len = (int)tmp.length() - 1;
            tmp=int_to_str_b(len);
        }

        cout << setw(3) << i << " | " << str << '\n';
    }
}

int main() {
    cout << "Fixed + variable\n";
    fix_var();
    cout << "Gamma\n";
    gamma(32);
    cout << endl << "Omega\n";
    omega(32);
}