#include "TFracEditor.h"

const char TFracEditor::divider = '/';
const std::string TFracEditor::zeroFrac = "0/1";

TFracEditor::TFracEditor() : frac(zeroFrac) {}

void TFracEditor::simplify() {
    unsigned int average = frac.find('/');
    if (average != -1) {
        char buf1[30], buf2[30];
        size_t n1 = frac.copy(buf1, average, 0);
        size_t n2 = frac.copy(buf2, frac.length() - average - 1, average + 1);
        buf1[n1] = '\0';
        buf2[n2] = '\0';
        int num = atoi(buf1);
        int den = atoi(buf2);

        int g = std::gcd(num, den);
        num /= g;
        den /= g;
        if (den < 0) {
            num *= -1;
            den *= -1;
        }
        frac = std::to_string(num) + "/" + std::to_string(den);
    }
}


bool TFracEditor::isZero() {
    return frac == zeroFrac;
}

std::string TFracEditor::addSign() {
    if (frac[0] == '-')
        frac.erase(frac.begin());
    else if (frac != zeroFrac)
        frac = "-" + frac;
    return frac;
}

std::string TFracEditor::addNumber(int num) {
    if (frac == zeroFrac)
        frac.erase();
    frac += std::to_string(num);
    simplify();
    return frac;
}

std::string TFracEditor::addZero() {
    return addNumber(0);
}

std::string TFracEditor::removeLastDigit() {
    frac.pop_back();
    if (frac == "-" || frac.empty())
        frac = zeroFrac;
    return frac;
}

std::string TFracEditor::clear() {
    return frac = zeroFrac;
}

std::string TFracEditor::addDivider() {
    if (frac.find(divider) == std::string::npos)
        frac += divider;
    else
        throw std::invalid_argument("Fraction already has a divider.");
    return frac;
}

void TFracEditor::editFraction(Operations operation) {
    switch (operation) {
        case Operations::ADD_SIGN:
            addSign();
            break;
        case Operations::ADD_DIGIT:
            int num;
            std::cout << "Enter number: ";
            std::cin >> num;
            addNumber(num);
            break;
        case Operations::ADD_ZERO:
            addZero();
            break;
        case Operations::REMOVE_LAST_DIGIT:
            removeLastDigit();
            break;
        case Operations::CLEAR:
            clear();
            break;
        case Operations::ADD_DIVIDER:
            addDivider();
            break;
        default:
            break;
    }
}

bool TFracEditor::isValid(std::string fraction) {
    bool isValid = false;
    if (!fraction.empty() && (fraction.find('/') != std::string::npos)) {
        unsigned int position = fraction.find('/');
        std::string firstPart = fraction;
        std::string secondPart = fraction;
        std::string numerator = firstPart.erase(position, fraction.length());
        std::string denominator = secondPart.erase(0, position);

        int digitsCounter = 0;
        for (char i : numerator) {
            if (isdigit(i))
                digitsCounter++;
        }

        for (char i : denominator) {
            if (isdigit(i))
                digitsCounter++;
        }

        if (digitsCounter != 0) {
            if (fraction[0] == '-') {
                if (digitsCounter == fraction.length() - 2)
                    isValid = true;
            } else {
                if (digitsCounter == fraction.length() - 1)
                    isValid = true;
            }
        }
    }
    return isValid;
}

void TFracEditor::setFraction(std::string fraction) {
    if (isValid(fraction)) {
        frac = fraction;
        simplify();
    } else {
        throw std::invalid_argument("Invalid fraction!");
    }
}

std::string TFracEditor::getFraction() {
    return frac;
}
