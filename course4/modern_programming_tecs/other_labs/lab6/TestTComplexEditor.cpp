#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"

#include "TComplexEditor.h"


TEST_SUITE ("TComplexEditorTest") {

    TEST_CASE ("TComplexEditor_construction") {
        std::string expected = "0";
        TComplexEditor a;
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_isZero") {
        TComplexEditor a;
                CHECK(a.isZero());
    }

    TEST_CASE ("TComplexEditor_isZero_2") {
        TComplexEditor a;
        a.addNumber(1);
                CHECK_FALSE(a.isZero());
    }

    TEST_CASE ("TComplexEditor_addNumber") {
        std::string expected = "1";
        TComplexEditor a;
        a.addNumber(1);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addNumber2") {
        std::string expected = "-16";
        TComplexEditor a;
        a.addNumber(-16);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addSign") {
        std::string expected = "-16";
        TComplexEditor a;
        a.addNumber(16);
        a.addSign();
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addSign2") {
        std::string expected = "16+i*12";
        TComplexEditor a;
        a.setComplexString("-16+i*12");
        a.addSign();
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addSign3") {
        std::string expected = "-5-i*10";
        TComplexEditor a;
        a.addNumber(-5);
        a.addImPart();
        a.addNumber(10);
        a.addSign();
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addImPart") {
        std::string expected = "15+i*4";
        TComplexEditor a;
        a.addNumber(15);
        a.addImPart();
        a.addNumber(4);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addImPart2") {
        std::string expected = "-5+i*10";
        TComplexEditor a;
        a.addNumber(-5);
        a.addImPart();
        a.addNumber(10);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_addImPart3") {
        TComplexEditor a;
        a.addImPart();
                CHECK_THROWS_AS(a.addImPart(), std::logic_error);
    }

    TEST_CASE ("TComplexEditor_editComplex") {
        std::string expected = "-8";
        TComplexEditor a;
        a.addNumber(8);
        a.editComplex(ComplexOperations::ADD_SIGN);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_editComplex2") {
        std::string expected = "8";
        TComplexEditor a;
        a.addNumber(-8);
        a.editComplex(ComplexOperations::ADD_SIGN);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_editComplex3") {
        std::string expected = "8+i*4";
        TComplexEditor a;
        a.addNumber(8);
        a.editComplex(ComplexOperations::ADD_IMPART);
        a.addNumber(4);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_editComplex4") {
        std::string expected = "8+i*40";
        TComplexEditor a;
        a.addNumber(8);
        a.editComplex(ComplexOperations::ADD_IMPART);
        a.addNumber(4);
        a.editComplex(ComplexOperations::ADD_ZERO);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_editComplex5") {
        std::string expected = "8-i*4";
        TComplexEditor a;
        a.addNumber(8);
        a.editComplex(ComplexOperations::ADD_IMPART);
        a.addNumber(4);
        a.editComplex(ComplexOperations::ADD_SIGN);
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_setComplexString") {
        std::string expected = "8+i*4";
        TComplexEditor a;
        a.setComplexString("8+i*4");
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_setComplexString2") {
        std::string expected = "-8-i*4";
        TComplexEditor a;
        a.setComplexString("-8-i*4");
                CHECK(expected == a.getComplexString());
    }

    TEST_CASE ("TComplexEditor_setComplexString3") {
        TComplexEditor a;
                CHECK_THROWS_AS(a.setComplexString("i*4"), std::invalid_argument);
    }
}
