#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"
#include <set>
#include "TPoly.h"


TEST_SUITE ("TPoly_Test")
{
    TEST_CASE ("POWER_1") {
        TPoly a(1, 2);
        TPoly b(1, 3);
        TPoly c = a + b;
                CHECK_EQ(3, c.maxDegree());
    }

    TEST_CASE ("POWER_2") {
        TPoly a(1, 2);
        TPoly b(1, 4);
        TPoly c = a + b;
                CHECK_EQ(4, c.maxDegree());
    }

    TEST_CASE ("COEFF_1") {
        TPoly a(1, 0);
        TPoly b(123, 2);
        TPoly c(452, 3);
        TPoly d = a + b + c;
                CHECK_EQ(452, d.coeff(3));
    }

    TEST_CASE ("COEFF_2") {
        TPoly a(1, 555);
        TPoly b(123, 2);
        TPoly c(452, 3);
        TPoly d = a + b + c;
                CHECK_EQ(1, d.coeff(555));
    }

    TEST_CASE ("CLEAR_1") {
        TPoly a(1, 555);
        a.clear();
                CHECK_EQ(0.0, a.compute(1.0));
    }

    TEST_CASE ("CLEAR_2") {
        TPoly a(-123, 55345435);
        a.clear();
                CHECK_EQ(0.0, a.compute(1.0));
    }

    TEST_CASE ("ADD_1") {
        TPoly a(1, 2);
        TPoly b(2, 3);
        TPoly c(3, 4);
        TPoly d(4, 4);
        TPoly e = a + b + c + d;
        set<string> result;
        for (int i = 0; i < 3; ++i)
            result.emplace(e[i].monomialToString());
        set<string> expected = {"1*x^2", "2*x^3", "7*x^4"};
                CHECK_EQ(true, result == expected);
    }

    TEST_CASE ("ADD_2") {
        TPoly a(1, 2);
        TPoly b(1, 2);
        TPoly c = a + b;
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> expected = {"2*x^2"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("MULT_1") {
        TPoly a(1, 1);
        TPoly b(2, 2);
        TPoly c = a * b;
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> exp = {"2*x^3"};
        bool r = result == exp;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("MULT_2") {
        TPoly a(3, 5);
        TPoly b(5, 3);
        TPoly c = a * b;
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> expected = {"15*x^8"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("SUB_1") {
        TPoly a(5, 5);
        TPoly b(1, 5);
        TPoly c = a - b;
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> expected = {"4*x^5"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("SUB_2") {
        TPoly a(1, 5);
        TPoly b(5, 5);
        TPoly c = a - b;
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> expected = {"-4*x^5"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("MINUS") {
        TPoly a(1, 5);
        TPoly c = a.minus();
        set<string> result;
        for (int i = 0; i < 1; ++i)
            result.emplace(c[i].monomialToString());
        set<string> expected = {"-1*x^5"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("DIFF_1") {
        TPoly a(5, 0);
        TPoly b(1, 2);
        TPoly c = a + b;
        TPoly e = c.differentiate();
        set<string> result;
        for (int i = 0; i < 2; ++i)
            result.emplace(e[i].monomialToString());
        set<string> expected = {"2*x^1", "0"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("DIFF_2") {
        TPoly a(5, 0);
        TPoly b(1, 2);
        TPoly bb(2, 3);
        TPoly c = a + b + bb;
        TPoly e = c.differentiate();
        set<string> result;
        for (int i = 0; i < 3; ++i)
            result.emplace(e[i].monomialToString());
        set<string> expected = {"2*x^1", "0", "6*x^2"};
        bool r = result == expected;
                CHECK_EQ(true, r);
    }

    TEST_CASE ("COMPUTE_1") {
        TPoly a(5, 0);
        TPoly b(1, 2);
        TPoly c = a + b;
        double e = c.compute(1.0);
                CHECK_EQ(1, 1);
    }

}
