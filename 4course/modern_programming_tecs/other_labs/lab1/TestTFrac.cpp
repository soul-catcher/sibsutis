#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"
#include "TFrac.h"

TEST_SUITE ("constructors") {
    TEST_CASE ("int constructor") {
        auto frac = TFrac(5, 6);
        CHECK(frac.getDividend() == 5);
        CHECK(frac.getDivisor() == 6);
    }

    TEST_CASE ("zero divisor") {
        CHECK_THROWS_AS(TFrac(4, 0), ZeroDivisionError);
    }

    TEST_CASE("simple str constructor") {
        auto frac = TFrac("5/6");
        CHECK(frac.getDividend() == 5);
        CHECK(frac.getDivisor() == 6);
    }

    TEST_CASE("negative dividend") {
        auto frac = TFrac("-5/6");
        CHECK(frac.getDividend() == -5);
        CHECK(frac.getDivisor() == 6);
    }

    TEST_CASE("negative divisor") {
        auto frac = TFrac("5/-6");
        CHECK(frac.getDividend() == -5);
        CHECK(frac.getDivisor() == 6);
    }

    TEST_CASE("copy constructor") {
        auto frac = TFrac(5, 6);
        auto frac2 = TFrac(frac);
        CHECK(frac2.getDividend() == 5);
        CHECK(frac2.getDivisor() == 6);
    }
}

TEST_SUITE("math") {
    TEST_CASE("add") {
        CHECK(TFrac(3, 5) + TFrac(4, 6) == TFrac(19, 15));
    }

    TEST_CASE("multiply") {
        CHECK(TFrac(2, 3) * TFrac(1, 5) == TFrac(2, 15));
    }

    TEST_CASE("sub") {
        CHECK(TFrac(3, 5) - TFrac(1, 12) == TFrac(31, 60));
    }

    TEST_CASE("sub zero") {
        CHECK(TFrac(3, 5) - TFrac(3, 5) == TFrac(0, 1));
    }

    TEST_CASE("divide") {
        CHECK(TFrac(3, 5) / TFrac(2, 3) == TFrac(9, 10));
    }

    TEST_CASE("square") {
        CHECK(TFrac(3, 4).square() == TFrac(9, 16));
    }

    TEST_CASE("inverse") {
        CHECK(TFrac(5, 3).inverse() == TFrac(3, 5));
    }

    TEST_CASE("negative") {
        CHECK(-TFrac(3, 5) == TFrac(-3, 5));
    }

    TEST_CASE("equal") {
        CHECK(TFrac(2, 3) == TFrac(6, 9));
    }

    TEST_CASE("not equal with ==") {
        CHECK_FALSE(TFrac(2, 3) == TFrac(6, 8));
    }

    TEST_CASE("not equal with !=") {
        CHECK(TFrac(2, 3) != TFrac(6, 8));
    }

    TEST_CASE("greater") {
        CHECK(TFrac(4, 5) > TFrac(3, 4));
    }
}

TEST_SUITE("getters") {
    TEST_CASE("dividend") {
        auto frac = TFrac(14, 11);
        CHECK(std::to_string(frac.getDividend()) == frac.getStrDividend());
    }

    TEST_CASE("divisor") {
        auto frac = TFrac(14, 11);
        CHECK(std::to_string(frac.getDivisor()) == frac.getStrDivisor());
    }
    TEST_CASE("to string stream") {
        std::stringstream ss;
        ss << TFrac(-12, 17);
        CHECK(ss.str() == "-12/17");
    }
}

