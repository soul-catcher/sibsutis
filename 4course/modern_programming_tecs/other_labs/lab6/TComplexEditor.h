#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>

enum class ComplexOperations {
    ADD_SIGN,
    ADD_DIGIT,
    ADD_ZERO,
    ADD_IMPART,
    REMOVE_LAST_DIGIT,
    CLEAR
};

enum class ComplexPart {
    REAL_PART,
    IM_PART
};

class TComplexEditor {
private:
    std::string complex;
    ComplexPart currentPart;
public:
    static const std::string zeroComplex;
    static const std::string imPart;

    TComplexEditor();

    bool isZero();

    std::string addSign();

    std::string addNumber(int a);

    std::string addZero();

    std::string addImPart();

    std::string removeLastDigit();

    std::string clear();

    void editComplex(ComplexOperations operation);

    std::string getComplexString();

    std::string setComplexString(std::string str);

    bool isValid(std::string);
};

