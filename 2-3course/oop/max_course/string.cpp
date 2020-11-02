#include "string.h"

String::String(const char str[]) : SequenceContainer(str){}

String::String() : SequenceContainer(){}

String &String::operator=(const char *arr) {
    this->arr = arr;
    return *this;
}

std::ostream &operator<<(std::ostream &out, const String &str) {
    if (str.arr) {
        out << str.arr;
    }
    return out;
}
