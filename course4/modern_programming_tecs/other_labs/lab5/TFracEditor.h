#pragma once

#include <iostream>
#include <string>
#include <numeric>
#include <exception>

enum class Operations {
    ADD_SIGN,
    ADD_DIGIT,
    ADD_ZERO,
    REMOVE_LAST_DIGIT,
    CLEAR,
    ADD_DIVIDER
};

class TFracEditor {
private:
    std::string frac;
public:
    static const char divider;
    static const std::string zeroFrac;

    TFracEditor();

    void simplify();

    bool isZero();

    std::string addSign();

    std::string addNumber(int num);

    std::string addZero();

    std::string removeLastDigit();

    std::string clear();

    std::string addDivider();

    void editFraction(Operations operation);

    void setFraction(std::string frac);

    std::string getFraction();

    bool isValid(std::string);
};

