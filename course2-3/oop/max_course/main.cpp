#include <iostream>
#include "string.h"

using namespace std;

int main() {
    String str("Constructor");
    cout << str << '\n';
    str.clear();
    cout << str.empty() << '\n';
    str = "assignment";
    cout << str << '\n';
    cout << str.empty() << '\n';
    cout << str[2] << endl;
}
