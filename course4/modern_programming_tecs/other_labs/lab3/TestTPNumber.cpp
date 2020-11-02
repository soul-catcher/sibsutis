#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"

#include "TPNumber.h"


TEST_SUITE ("TPNumberTest") {
    TEST_CASE ("CONSTR_1") {
                CHECK(10 == TPNumber(1, 10, 2).getBase());
    }

    TEST_CASE ("CONSTR_2") {
                CHECK(10 == TPNumber(1, -1, 2).getBase());
    }

    TEST_CASE ("CONSTR_STR_1") {
                CHECK(10 == TPNumber("1", "10", "2").getBase());
    }

    TEST_CASE ("COPY_1") {
        TPNumber a(1, 10, 2);
        TPNumber b = a.copy();
                CHECK(1.0 == b.getNumber());
    }

    TEST_CASE ("COPY_2") {
        TPNumber a(-1.0, 10, 2);
        TPNumber b = a.copy();
                CHECK(-1.0 == b.getNumber());
    }

    TEST_CASE ("ADD_1") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(5.0, 10, 2);
                CHECK(10.0 == a.add(b).getNumber());
    }

    TEST_CASE ("ADD_2") {
        TPNumber a(-5.0, 10, 2);
        TPNumber b(-5.0, 10, 2);
                CHECK(-10.0 == a.add(b).getNumber());
    }

    TEST_CASE ("MULT_1") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(5.0, 10, 2);
                CHECK(25.0 == a.mult(b).getNumber());
    }

    TEST_CASE ("MULT_2") {
        TPNumber a(-5.0, 10, 2);
        TPNumber b(-5.0, 10, 2);
                CHECK(25.0 == a.mult(b).getNumber());
    }

    TEST_CASE ("SUBTR_1") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(5.0, 10, 2);
                CHECK(0.0 == a.subtr(b).getNumber());
    }

    TEST_CASE ("SUBTR_2") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(-5.0, 10, 2);
                CHECK(10.0 == a.subtr(b).getNumber());
    }

    TEST_CASE ("DIV_1") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(5.0, 10, 2);
                CHECK(1.0 == a.div(b).getNumber());
    }

    TEST_CASE ("DIV_2") {
        TPNumber a(5.0, 10, 2);
        TPNumber b(-5.0, 10, 2);
                CHECK(-1.0 == a.div(b).getNumber());
    }

    TEST_CASE ("INV_1") {
        TPNumber a(5.0, 10, 3);
                CHECK(0.2 == a.inverse().getNumber());
    }

    TEST_CASE ("INV_2") {
        TPNumber a(1.0, 10, 2);
                CHECK(1.0 == a.inverse().getNumber());
    }

    TEST_CASE ("SQ_1") {
        TPNumber a(2.0, 10, 2);
                CHECK(4.0 == a.square().getNumber());
    }

    TEST_CASE ("SQ_2") {
        TPNumber a(1.0, 10, 2);
                CHECK(1.0 == a.square().getNumber());
    }

    TEST_CASE ("GETNUM_1") {
        TPNumber a(2.0, 10, 2);
                CHECK(2.0 == a.getNumber());
    }

    TEST_CASE ("GETNUM_2") {
        TPNumber a(-5.0, 10, 2);
                CHECK(-5.0 == a.getNumber());
    }

    TEST_CASE ("GETNUMST_1") {
        TPNumber a(15, 16, 2);
                CHECK("F" == a.getNumberString());
    }

    TEST_CASE ("GETNUMST_2") {
        TPNumber a(10, 16, 2);
                CHECK("A" == a.getNumberString());
    }

    TEST_CASE ("GETBASE_1") {
        TPNumber a(10, 16, 2);
                CHECK(16 == a.getBase());
    }

    TEST_CASE ("GETBASESTR_1") {
        TPNumber a(10, 16, 2);
                CHECK("16" == a.getBaseString());
    }

    TEST_CASE ("GETPR_1") {
        TPNumber a(10, 16, 2);
                CHECK(2 == a.getPrecision());
    }

    TEST_CASE ("GETPRSTR_1") {
        TPNumber a(10, 16, 2);
                CHECK("2" == a.getPrecisionString());
    }

    TEST_CASE ("SET_BASE_1") {
        TPNumber a(10, 16, 2);
        a.setBase(5);
                CHECK(5 == a.getBase());
    }

    TEST_CASE ("SET_BASE_STR_1") {
        TPNumber a(10, 16, 2);
        a.setBase("5");
                CHECK(5 == a.getBase());
    }

    TEST_CASE ("SET_PR_1") {
        TPNumber a(10, 16, 2);
        a.setPrecision(3);
                CHECK(3 == a.getPrecision());
    }

    TEST_CASE ("SET_PR_STR_1") {
        TPNumber a(10, 16, 2);
        a.setPrecision("3");
                CHECK(3 == a.getPrecision());
    }

}

