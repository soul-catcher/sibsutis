#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"
#include "TMonomial.h"



TEST_SUITE ("TMonomial_Test")
{
    TEST_CASE ("INIT_1") {
        TMonomial a(1, 1);
                CHECK_EQ(string("1*x^1"),
                         a.monomialToString());
    }

    TEST_CASE ("INIT_2") {
        TMonomial a(555, 1);
                CHECK_EQ(string("555*x^1"),
                         a.monomialToString());
    }

    TEST_CASE ("INIT_3") {
        TMonomial a(1, 222);
                CHECK_EQ(string("1*x^222"),
                         a.monomialToString());
    }

    TEST_CASE ("INIT_4") {
        TMonomial a(123, 123);
                CHECK_EQ(string("123*x^123"),
                         a.monomialToString());
    }

    TEST_CASE ("INIT_5") {
        TMonomial a(-1, -1);
                CHECK_EQ(string("-1*x^-1"),
                         a.monomialToString());
    }

    TEST_CASE ("WRPOW_1") {
        TMonomial a(1, 1);
        a.writePower(5);
                CHECK_EQ(string("1*x^5"),
                         a.monomialToString());
    }

    TEST_CASE ("WRPOW_2") {
        TMonomial a(1, 1);
        a.writePower(100);
                CHECK_EQ(string("1*x^100"),
                         a.monomialToString());
    }

    TEST_CASE ("WRCOEFF_1") {
        TMonomial a(1, 1);
        a.writeCoeff(100);
                CHECK_EQ(string("100*x^1"),
                         a.monomialToString());
    }

    TEST_CASE ("WRCOEFF_2") {
        TMonomial a(1, 1);
        a.writeCoeff(555);
                CHECK_EQ(string("555*x^1"),
                         a.monomialToString());
    }

    TEST_CASE ("EQ_1") {
        bool result = TMonomial(1, 1).isEqual(TMonomial(1, 1));
                CHECK_EQ(true, result);
    }

    TEST_CASE ("EQ_2") {
        bool result = TMonomial(1, 2).isEqual(TMonomial(1, 1));
                CHECK_EQ(false, result);
    }

    TEST_CASE ("DIFF_1") {
        TMonomial a(2, 1);
        TMonomial b = a.differentiate();
                CHECK_EQ(string("2*x^0"), b.monomialToString());
    }

    TEST_CASE ("DIFF_2") {
        TMonomial a(2, 2);
        TMonomial b = a.differentiate();
                CHECK_EQ(string("4*x^1"), b.monomialToString());
    }

    TEST_CASE ("COMP_1") {
        TMonomial a(2, 2);
        double b = a.compute(1.0);
                CHECK_EQ(2.0, b);
    }

    TEST_CASE ("COMP_2") {
        TMonomial a(2, 2);
        double b = a.compute(2.0);
                CHECK_EQ(8.0, b);
    }

    TEST_CASE ("RPOW_1") {
        TMonomial a(2, 2);
        int b = a.readPower();
                CHECK_EQ(2, b);
    }

    TEST_CASE ("RPOW_2") {
        TMonomial a(2, 3);
        int b = a.readPower();
                CHECK_EQ(3, b);
    }

    TEST_CASE ("RCOEFF_1") {
        TMonomial a(2, 3);
        int b = a.readCoeff();
                CHECK_EQ(2, b);
    }

    TEST_CASE ("RCOEFF_2") {
        TMonomial a(5, 3);
        int b = a.readCoeff();
                CHECK_EQ(5, b);
    }

}

