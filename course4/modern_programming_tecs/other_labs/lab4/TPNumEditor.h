#pragma once

#include <string>

using namespace std;

enum action {
    _changeSign,
    _addDigit,
    _addZero,
    _backspace,
    _clear,
    _addDivider,
    _editNumber,
};

class TPNumEditor {
private:
    std::string str;
public:
    TPNumEditor();

    bool equalsZero();

    string changeSign();

    string addDigit(int digit);

    string addZero();

    string backspace();

    string clear();

    string addDivider();

    string getNumberString();

    void setNumber(int num);

    string editNumber(string str);

    string menu(action action);
};
