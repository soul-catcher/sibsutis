#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <set>
#include "../doctest.h"
#include "TSet.h"


TEST_SUITE ("TSet_Test")
{
    TEST_CASE ("INIT_1") {
        TSet<int> a;
        set<int> b{};
                CHECK_EQ(true, a.container == b);
    }

    TEST_CASE ("INIT_2") {
        TSet<int> a;
        set<int> b{1};
                CHECK_EQ(true, a.container != b);
    }

    TEST_CASE ("CLEAR_1") {
        TSet<int> a;
        a.insert_(1);
        a.clear();
        set<int> b{};
                CHECK_EQ(true, a.container == b);
    }

    TEST_CASE ("CLEAR_2") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.clear();
        set<int> b{};
                CHECK_EQ(true, a.container == b);
    }

    TEST_CASE ("INSERT_1") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        set<int> b{1, 2};
                CHECK_EQ(true, a.container == b);
    }

    TEST_CASE ("INSERT_2") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        set<int> b{1, 2, 3};
                CHECK_EQ(true, a.container == b);
    }

    TEST_CASE ("IS_EMPTY") {
        TSet<int> a;
                CHECK_EQ(true, a.count() == 0);
    }

    TEST_CASE ("CONTAINS_1") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
                CHECK_EQ(true, a.contains(1));
    }

    TEST_CASE ("CONTAINS_2") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
                CHECK_EQ(true, a.contains(2));
    }

    TEST_CASE ("CONTAINS_3") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
                CHECK_EQ(true, a.contains(3));
    }

    TEST_CASE ("UNION_1") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{1, 2, 3, 4};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(4);
        TSet<int> result = a.add(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("UNION_2") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{1, 2, 3};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(1);
        b.insert_(2);
        b.insert_(3);
        TSet<int> result = a.add(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("SUBTRACT_1") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{2, 3};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(1);
        TSet<int> result = a.subtract(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("SUBTRACT_2") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{1, 2, 3};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(4);
        TSet<int> result = a.subtract(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("MULTIPLY_1") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{1, 2, 3};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(1);
        b.insert_(2);
        b.insert_(3);
        TSet<int> result = a.multiply(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("MULTIPLY_2") {
        TSet<int> a;
        TSet<int> b;
        set<int> c{1};
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
        b.insert_(1);
        b.insert_(4);
        b.insert_(5);
        TSet<int> result = a.multiply(b);
                CHECK_EQ(true, result.container == c);
    }

    TEST_CASE ("COUNTER_1") {
        TSet<int> a;
        a.insert_(1);
        a.insert_(2);
        a.insert_(3);
                CHECK_EQ(3, a.count());
    }

    TEST_CASE ("COUNTER_2") {
        TSet<int> a;
                CHECK_EQ(0, a.count());
    }

    TEST_CASE ("ELEMENT_3") {
        TSet<int> a;
        a.insert_(5);
                CHECK_EQ(5, a.element(0));
    }
}
