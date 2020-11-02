#include <regex>
#include <iostream>
#include "TPNumEditor.h"

TPNumEditor::TPNumEditor() {
    this->str = string("0");
}

bool TPNumEditor::equalsZero() {
    return this->str == "0";
}

string TPNumEditor::changeSign() {
    if (this->str[0] == '-')
        this->str.erase(0, 1);
    else if (this->str != "0")
        this->str.insert(0, "-");
    return this->str;
}

string TPNumEditor::addDigit(int digit) {
    if (digit < 0 || digit > 16)
        return this->str;

    string symbols = "0123456789ABCDEF";
    if (equalsZero())
        return this->str = symbols[digit];
    this->str.push_back(symbols[digit]);
    return this->str;
}

string TPNumEditor::addZero() {
    return this->addDigit(0);
}

string TPNumEditor::backspace() {
    this->str.pop_back();
    if (this->str.empty() || this->str == "-")
        return this->addZero();
    if (this->str[str.size() - 1] == '.')
        this->str.pop_back();
    return this->str;
}

string TPNumEditor::clear() {
    return this->str = "0";
}

string TPNumEditor::addDivider() {
    if (this->str.find(".") == string::npos)
        this->str.append(".");
    return this->str;
}

string TPNumEditor::getNumberString() {
    return this->str;
}

void TPNumEditor::setNumber(int num) {
    this->str = to_string(num);
}

string TPNumEditor::editNumber(string input) {
    regex symbols("-?(0|[1-9A-F][0-9A-F]*).[0-9A-F]*");
    if (regex_match(input, symbols))
        return this->str = input;
    return this->str;
}

string TPNumEditor::menu(action action) {
    string input;
    int num;
    switch (action) {
        case _addDigit:
            cout << "Add digit: ";
            cin >> num;
            return this->addDigit(num);
            break;
        case _addZero:
            return this->addZero();
            break;
        case _backspace:
            return this->backspace();
            break;
        case _clear:
            return this->clear();
            break;
        case _addDivider:
            return this->addDivider();
            break;
        case _editNumber:
            cout << "editNumber: ";
            cin >> input;
            return this->editNumber(input);
            break;
        default:
            return this->str;
            break;
    }
    return this->str;
}

