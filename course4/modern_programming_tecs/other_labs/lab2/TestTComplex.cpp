#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"
#include "TComplex.h"

TEST_SUITE("constructors") {
    TEST_CASE("double constructor") {
        auto complex = TComplex(4.3, 6.1);
        CHECK(complex.getReal() == 4.3);
        CHECK(complex.getImaginary() == 6.1);
    }

    TEST_CASE("str constructor with decimals") {
        auto complex = TComplex("76+i*44");
        CHECK(complex.getReal() == 76);
        CHECK(complex.getImaginary() == 44);
    }

    TEST_CASE("str constructor with floating points") {
        auto complex = TComplex("7.45+i*3.99");
        CHECK(complex.getReal() == 7.45);
        CHECK(complex.getImaginary() == 3.99);
    }
}

TEST_SUITE("math") {
    TEST_CASE("square") {
        CHECK(TComplex(3, 4).square() == TComplex(-7, 24));
    }

    TEST_CASE("inverse") {
        CHECK(TComplex(3, 4).inverse() == TComplex(0.12, -0.16));
    }

    TEST_CASE("abs") {
        CHECK(TComplex(3, 4).abs() == hypot(3, 4));
    }

    TEST_CASE("angle rad") {
        CHECK(TComplex(1, 1).angle_rad() == doctest::Approx(0.79).epsilon(0.01));
    }

    TEST_CASE("angle deg") {
        CHECK(TComplex(1, 1).angle_deg() == 45);
    }

    TEST_CASE("add") {
        CHECK(TComplex(3, 4) + TComplex(2, 3) == TComplex(5, 7));
    }

    TEST_CASE("mul") {
        CHECK(TComplex(2, 3) * TComplex(2, 2) == TComplex(-2, 10));
    }

    TEST_CASE("sub") {
        CHECK(TComplex(3, 4) - TComplex(2, 3) == TComplex(1, 1));
    }
}

