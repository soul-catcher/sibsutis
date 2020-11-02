#pragma once

#include <regex>
#include <cmath>
#include <stdexcept>

class TComplex {
    double real, imaginary;

public:
    TComplex(double real, double imaginary) : real(real), imaginary(imaginary) {}

    explicit TComplex(const std::string &complex) {
        std::regex re(R"(^([-+]?\d+.?\d*)\s?\+\s?i\s?\*\s?([-+]?\d+.?\d*)$)");
        std::smatch matches;
        std::regex_search(complex, matches, re);
        if (matches.empty()) {
            throw std::invalid_argument(complex);
        }
        real = std::stod(matches[1].str());
        imaginary = std::stod(matches[2].str());
    }

    [[nodiscard]] double getReal() const {
        return real;
    }

    [[nodiscard]] double getImaginary() const {
        return imaginary;
    }

    [[nodiscard]] std::string getStrReal() const {
        return std::to_string(real);
    }

    [[nodiscard]] std::string getStrImaginary() const {
        return std::to_string(imaginary);
    }

    [[nodiscard]] TComplex square() const {
        return TComplex(real * real - imaginary * imaginary, real * imaginary * 2);
    }

    [[nodiscard]] TComplex inverse() const {
        return TComplex(real / (real * real + imaginary * imaginary),
                        -imaginary / (real * real + imaginary * imaginary));
    }

    [[nodiscard]] double abs() const {
        return hypot(real, imaginary);
    }

    [[nodiscard]] double angle_rad() const {
        if (real > 0) {
            return atan(imaginary / real);
        } else if (real == 0 && imaginary > 0) {
            return M_PI_2;
        } else if (real < 0) {
            return atan(imaginary / real) + M_PI;
        } else if (real == 0 && imaginary < 0) {
            return -M_PI_2;
        } else {
            throw std::domain_error("Cannot calculate angle_rad for 0+i*0");
        }
    }

    [[nodiscard]] double angle_deg() const {
        return angle_rad() * 180 / M_PI;
    }

    [[nodiscard]] TComplex power(int n) const {
        return TComplex(pow(abs(), n) * cos(angle_rad() * n), pow(abs(), n) * sin(angle_rad()) * n);
    }

    [[nodiscard]] TComplex root(int n, int i) const {
        return TComplex(pow(abs(), 1. / n) * cos((angle_rad() + 2 * M_PI * i) / n),
                        pow(abs(), 1. / n) * sin((angle_rad() + 2 * M_PI * i) / n));
    }
};

std::ostream &operator<<(std::ostream &ostream, const TComplex &complex) {
    return ostream << complex.getReal() << "+i*" << complex.getImaginary();
}

TComplex operator+(const TComplex &a, const TComplex &b) {
    return TComplex(a.getReal() + b.getReal(), a.getImaginary() + b.getImaginary());
}

TComplex operator*(const TComplex &a, const TComplex &b) {
    return TComplex(a.getReal() * b.getReal() - a.getImaginary() * b.getImaginary(),
                    a.getReal() * b.getImaginary() + b.getReal() * a.getImaginary());
}

TComplex operator-(const TComplex &a, const TComplex &b) {
    return TComplex(a.getReal() - b.getReal(), a.getImaginary() - b.getImaginary());
}

TComplex operator/(const TComplex &a, const TComplex &b) {
    return TComplex((a.getReal() * b.getReal() + a.getImaginary() * b.getImaginary()) /
                    (b.getReal() * b.getReal() + b.getImaginary() * b.getImaginary()),
                    (b.getReal() * a.getImaginary() - a.getReal() * b.getImaginary()) /
                    (b.getReal() * b.getReal() + b.getImaginary() * b.getImaginary()));
}

TComplex operator-(const TComplex &a) {
    return TComplex(-a.getReal(), -a.getImaginary());
}

bool operator==(const TComplex &a, const TComplex &b) {
    return a.getReal() == b.getReal() && a.getImaginary() == b.getImaginary();
}

bool operator!=(const TComplex &a, const TComplex &b) {
    return a.getReal() != b.getReal() || a.getImaginary() != b.getImaginary();
}

