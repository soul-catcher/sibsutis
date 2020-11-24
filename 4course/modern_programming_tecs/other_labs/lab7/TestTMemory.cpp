#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"

#include "TMemory.h"
#include "../lab1/TFrac.h"


TEST_SUITE ("TMemory_Test") {
    TEST_CASE ("ADD_1") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
        tmem.add(tfrac);
                CHECK(string("2/5") == tmem.get().getFractionString());
    }

    TEST_CASE ("ADD_2") {
        TFrac tfrac(1, 4);
        TMemory<TFrac> tmem(tfrac);
        tmem.add(tfrac);
                CHECK(string("1/2") == tmem.get().getFractionString());
    }

    TEST_CASE ("ADD_3") {
        TFrac tfrac(1, 999);
        TMemory<TFrac> tmem(tfrac);
        tmem.add(tfrac);
                CHECK(string("2/999") == tmem.get().getFractionString());
    }

    TEST_CASE ("WRITE_1") {
        TFrac tfrac(1, 5);
        TFrac tfrac2(5, 7);
        TMemory<TFrac> tmem(tfrac);
        tmem.write(tfrac2);
                CHECK(string("5/7") == tmem.get().getFractionString());
    }

    TEST_CASE ("WRITE_2") {
        TFrac tfrac(1, 5);
        TFrac tfrac2(1, 10);
        TMemory<TFrac> tmem(tfrac);
        tmem.write(tfrac2);
                CHECK(string("1/10") == tmem.get().getFractionString());
    }

    TEST_CASE ("WRITE_3") {
        TFrac tfrac(1, 5);
        TFrac tfrac2(2, 99);
        TMemory<TFrac> tmem(tfrac);
        tmem.write(tfrac2);
                CHECK(string("2/99") == tmem.get().getFractionString());
    }

    TEST_CASE ("GET_1") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(string("1/5") == tmem.get().getFractionString());
    }

    TEST_CASE ("GET_2") {
        TFrac tfrac(2, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(string("2/5") == tmem.get().getFractionString());
    }

    TEST_CASE ("GET_3") {
        TFrac tfrac(3, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(string("3/5") == tmem.get().getFractionString());
    }

    TEST_CASE ("CLEAR_1") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
        tmem.clear();
                CHECK(string("0/1") == tmem.get().getFractionString());
    }

    TEST_CASE ("READ_FSTATE_1") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
        tmem.clear();
                CHECK(string("0") == tmem.readFState());
    }

    TEST_CASE ("READ_FSTATE_2") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
        tmem.add(tfrac);
                CHECK(string("1") == tmem.readFState());
    }

    TEST_CASE ("READ_NUMBER_1") {
        TFrac tfrac(1, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(tfrac.getFractionString() == tmem.readNumber().getFractionString());
    }

    TEST_CASE ("READ_NUMBER_2") {
        TFrac tfrac(2, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(tfrac.getFractionString() == tmem.readNumber().getFractionString());
    }

    TEST_CASE ("READ_NUMBER_3") {
        TFrac tfrac(4, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(tfrac.getFractionString() == tmem.readNumber().getFractionString());
    }

    TEST_CASE ("READ_NUMBER_4") {
        TFrac tfrac(5, 5);
        TMemory<TFrac> tmem(tfrac);
                CHECK(tfrac.getFractionString() == tmem.readNumber().getFractionString());
    }
}
