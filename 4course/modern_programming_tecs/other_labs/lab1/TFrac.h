#pragma once

#include <stdexcept>
#include <numeric>

class ZeroDivisionError : std::runtime_error {
public:
    explicit ZeroDivisionError(int dividend) : runtime_error(
            "Attempted to divide " + std::to_string(dividend) + " by zero") {}
};

class TFrac {
    int a, b;

public:
    TFrac() : a(0), b(1) {}

    TFrac(int a, int b) : a(a), b(b) {
        check_divisor_by_zero();
        reduce();
    }

    explicit TFrac(const std::string &fraction) {
        std::size_t slash_ind;
        a = std::stoi(fraction, &slash_ind);
        b = std::stoi(&fraction[slash_ind + 1]);
        check_divisor_by_zero();
        reduce();
    }

    [[nodiscard]] std::string getFractionString() const {
        return std::to_string(a) + '/' + std::to_string(b);
    }

    [[nodiscard]] int getDividend() const {
        return a;
    }

    [[nodiscard]] int getDivisor() const {
        return b;
    }

    [[nodiscard]] std::string getStrDividend() const {
        return std::to_string(a);
    }

    [[nodiscard]] std::string getStrDivisor() const {
        return std::to_string(b);
    }

    [[nodiscard]] TFrac square() const {
        return TFrac(a * a, b * b);
    }

    [[nodiscard]] TFrac inverse() const {
        return TFrac(b, a);
    }

private:
    void reduce() {
        if (b < 0) {
            a *= -1;
            b *= -1;
        }
        auto gcd = std::gcd(a, b);
        a /= gcd;
        b /= gcd;
    }

    void check_divisor_by_zero() const {
        if (b == 0) {
            throw ZeroDivisionError(a);
        }
    }
};

std::ostream &operator<<(std::ostream &ostream, const TFrac &frac) {
    return ostream << frac.getFractionString();
}

TFrac operator+(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return TFrac(lcm / a.getDivisor() * a.getDividend() + lcm / b.getDivisor() * b.getDividend(), lcm);
}

TFrac operator*(const TFrac &a, const TFrac &b) {
    return TFrac(a.getDividend() * b.getDividend(), a.getDivisor() * b.getDivisor());
}

TFrac operator-(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return TFrac(lcm / a.getDivisor() * a.getDividend() - lcm / b.getDivisor() * b.getDividend(), lcm);
}

TFrac operator/(const TFrac &a, const TFrac &b) {
    return TFrac(a.getDividend() * b.getDivisor(), a.getDivisor() * b.getDividend());
}

TFrac operator-(const TFrac &a) {
    return TFrac(-a.getDividend(), a.getDivisor());
}

bool operator==(const TFrac &a, const TFrac &b) {
    return a.getDividend() == b.getDividend() && a.getDivisor() == b.getDivisor();
}

bool operator!=(const TFrac &a, const TFrac &b) {
    return a.getDividend() != b.getDividend() || a.getDivisor() != b.getDivisor();
}

bool operator>(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return lcm / a.getDivisor() * a.getDividend() > lcm / b.getDivisor() * b.getDividend();
}

bool operator>=(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return lcm / a.getDivisor() * a.getDividend() >= lcm / b.getDivisor() * b.getDividend();
}

bool operator<(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return lcm / a.getDivisor() * a.getDividend() < lcm / b.getDivisor() * b.getDividend();
}

bool operator<=(const TFrac &a, const TFrac &b) {
    auto lcm = std::lcm(a.getDivisor(), b.getDivisor());
    return lcm / a.getDivisor() * a.getDividend() <= lcm / b.getDivisor() * b.getDividend();
}

