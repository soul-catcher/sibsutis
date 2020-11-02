#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"
#include "lab0_3.h"

TEST_SUITE ("max3") {
    TEST_CASE ("test1") {
        CHECK(max3(0, 0, 0) == 0);
    }
}

TEST_SUITE ("task2") {
    TEST_CASE ("test1") {
        CHECK(task2(0) == 0);
    }
    TEST_CASE ("test2") {
        CHECK(task2(10) == 1);
    }
}

TEST_SUITE ("min_digit") {
    TEST_CASE ("test1") {
        CHECK(min_digit(0) == 0);
    }
}

TEST_SUITE ("task4") {
    TEST_CASE ("test1") {
        CHECK(task4({}) == 0);
    }
    TEST_CASE ("test2") {
        CHECK(task4({{}, {}}) == 0);
    }
    TEST_CASE ("test3") {
        CHECK(task4({{}, {0}}) == 0);
    }
    TEST_CASE ("test4") {
        CHECK(task4({{}, {1}}) == 1);
    }
}

