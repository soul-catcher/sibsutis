#include "TComplexEditor.h"

const std::string TComplexEditor::zeroComplex = "0";
const std::string TComplexEditor::imPart = "+i*";

TComplexEditor::TComplexEditor() : complex(zeroComplex), currentPart(ComplexPart::REAL_PART) {}

bool TComplexEditor::isZero() {
    return complex == zeroComplex;
}

std::string TComplexEditor::addSign() {
    if (currentPart == ComplexPart::REAL_PART) {
        if (complex[0] == '-')
            complex.erase(complex.begin());
        else if (complex[0] != '0')
            complex = '-' + complex;
    } else {
        auto pos = complex.find('i');
        if (complex[pos - 1] == '-')
            complex[pos - 1] = '+';
        else
            complex[pos - 1] = '-';
    }
    return complex;
}

std::string TComplexEditor::addNumber(int a) {
    if (currentPart == ComplexPart::IM_PART)
        complex += std::to_string(a);
    else {
        auto pos = complex.find('i');
        if (complex[0] == '0')
            complex.replace(0, 1, std::to_string(a));
        else if (pos != -1)
            complex.insert(pos - 1, std::to_string(a));
        else if (complex[0] == '-')
            complex.insert(1, std::to_string(a));
        else
            complex += std::to_string(a);
    }
    return complex;
}

std::string TComplexEditor::addZero() {
    return addNumber(0);
}

std::string TComplexEditor::addImPart() {
    if (complex.find(imPart) == std::string::npos) {
        complex = complex + imPart;
        currentPart = ComplexPart::IM_PART;
    } else {
        throw std::logic_error("Complex number already has a image part");
    }
    return complex;
}

std::string TComplexEditor::removeLastDigit() {
    complex.pop_back();
    if (currentPart == ComplexPart::REAL_PART) {
        if (complex == "-" || complex.empty())
            complex = zeroComplex;
    } else {
        if (complex[complex.length() - 1] == '*') {
            complex.erase(complex.end() - 3, complex.end());
            currentPart = ComplexPart::REAL_PART;
        } else if (complex.find('i'))
            complex.back();
    }
    return complex;
}

std::string TComplexEditor::clear() {
    currentPart = ComplexPart::REAL_PART;
    return complex = zeroComplex;
}

void TComplexEditor::editComplex(ComplexOperations operation) {
    switch (operation) {
        case ComplexOperations::ADD_SIGN:
            addSign();
            break;
        case ComplexOperations::ADD_DIGIT:
            int num;
            std::cout << "Enter number to add: ";
            std::cin >> num;
            addNumber(num);
            break;
        case ComplexOperations::ADD_ZERO:
            addZero();
            break;
        case ComplexOperations::ADD_IMPART:
            addImPart();
            break;
        case ComplexOperations::REMOVE_LAST_DIGIT:
            removeLastDigit();
            break;
        case ComplexOperations::CLEAR:
            clear();
            break;
        default:
            break;
    }
}

std::string TComplexEditor::getComplexString() {
    return complex;
}

bool TComplexEditor::isValid(std::string str) {
    bool res = false;
    if (!str.empty() && (str.find("+i*") != std::string::npos || str.find("-i*") != std::string::npos)) {
        auto pos = str.find('i');
        std::string tempRe = str;
        std::string tempIm = str;
        std::string rePart = tempRe.erase(pos - 1, str.length());
        std::string imPart = tempIm.erase(0, pos + 2);

        int digitCount = 0;
        for (auto i : rePart) {
            if (isdigit(i))
                digitCount++;
        }
        for (auto i : imPart) {
            if (isdigit(i))
                digitCount++;
        }

        if (digitCount) {
            if (str[0] == '-') {
                if (digitCount == str.length() - 4)
                    res = true;
            } else {
                if (digitCount == str.length() - 3)
                    res = true;
            }
        }
    }
    return res;
}

std::string TComplexEditor::setComplexString(std::string str) {
    if (isValid(str))
        complex = str;
    else
        throw std::invalid_argument("Wrong complex number!");
    return std::string();
}
