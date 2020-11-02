#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"

#include "TFracEditor.h"


TEST_SUITE ("TFracEditorTest") {
    TEST_CASE ("FracEditor_Constructor") {
        std::string expected = "0/1";
        TFracEditor editor;
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_AddNumber") {
        std::string expected = "8";
        TFracEditor editor;
        editor.addNumber(8);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_AddNumber_2") {
        std::string expected = "-8";
        TFracEditor editor;
        editor.addNumber(-8);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_isZero") {
        TFracEditor editor;
                CHECK(editor.isZero());
    }

    TEST_CASE ("FracEditor_isZero_2") {
        TFracEditor editor;
        editor.addNumber(8);
                CHECK_FALSE(editor.isZero());
    }

    TEST_CASE ("FracEditor_addSign") {
        std::string expected = "-8";
        TFracEditor editor;
        editor.addNumber(8);
        editor.addSign();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_addSign_2") {
        std::string expected = "8";
        TFracEditor editor;
        editor.addNumber(-8);
        editor.addSign();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_addSign_3") {
        std::string expected = "-1/2";
        TFracEditor editor;
        editor.addNumber(8);
        editor.addDivider();
        editor.addNumber(16);
        editor.addSign();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_addZero") {
        std::string expected = "0";
        TFracEditor editor;
        editor.addZero();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_addZero_2") {
        std::string expected = "80";
        TFracEditor editor;
        editor.addNumber(8);
        editor.addZero();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_removeLastDigit") {
        std::string expected = "8";
        TFracEditor editor;
        editor.addNumber(80);
        editor.removeLastDigit();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_removeLastDigit_2") {
        std::string expected = "-8";
        TFracEditor editor;
        editor.addNumber(-80);
        editor.removeLastDigit();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_removeLastDigit_3") {
        std::string expected = "-3/";
        TFracEditor editor;
        editor.addNumber(3);
        editor.addDivider();
        editor.addNumber(-7);
        editor.removeLastDigit();
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_editFraction") {
        std::string expected = "0";
        TFracEditor editor;
        editor.editFraction(Operations::ADD_ZERO);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_editFraction_2") {
        std::string expected = "1/2";
        TFracEditor editor;
        editor.addNumber(8);
        editor.editFraction(Operations::ADD_DIVIDER);
        editor.addNumber(16);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_editFraction_3") {
        std::string expected = "1/";
        TFracEditor editor;
        editor.addNumber(8);
        editor.editFraction(Operations::ADD_DIVIDER);
        editor.addNumber(16);
        editor.editFraction(Operations::REMOVE_LAST_DIGIT);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_editFraction_4") {
        std::string expected = "-1/2";
        TFracEditor editor;
        editor.addNumber(8);
        editor.editFraction(Operations::ADD_DIVIDER);
        editor.addNumber(16);
        editor.editFraction(Operations::ADD_SIGN);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_editFraction_5") {
        std::string expected = "0/1";
        TFracEditor editor;
        editor.addNumber(8);
        editor.editFraction(Operations::ADD_DIVIDER);
        editor.addNumber(16);
        editor.editFraction(Operations::CLEAR);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_getFraction") {
        std::string expected = "0/1";
        TFracEditor editor;
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_getFraction_2") {
        std::string expected = "8";
        TFracEditor editor;
        editor.addNumber(8);
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_setFraction") {
        std::string expected = "5/2";
        TFracEditor editor;
        editor.setFraction("5/2");
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_setFraction_2") {
        std::string expected = "-5/2";
        TFracEditor editor;
        editor.setFraction("-5/2");
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_setFraction_3") {
        std::string expected = "-5/2";
        TFracEditor editor;
        editor.setFraction("-10/4");
                CHECK(expected == editor.getFraction());
    }

    TEST_CASE ("FracEditor_setFraction_4") {
        TFracEditor editor;
                CHECK_THROWS_AS(editor.setFraction("1"), std::invalid_argument);
    }
}

