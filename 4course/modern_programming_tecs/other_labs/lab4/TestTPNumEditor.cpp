#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

#include "TPNumEditor.h"


TEST_SUITE ("TPNumEditorTest") {

    TEST_CASE ("CONSTR_1") {
                CHECK("0" == TPNumEditor().getNumberString());
    }

    TEST_CASE ("EQ_ZERO_1") {
                CHECK(true == TPNumEditor().equalsZero());
    }

    TEST_CASE ("EQ_ZERO_2") {
        TPNumEditor tpneditor;
        tpneditor.addDigit(5);
                CHECK(false == tpneditor.equalsZero());
    }

    TEST_CASE ("CHANGE_SIGN_1") {
        TPNumEditor o1;
        o1.addDigit(5);
                CHECK("-5" == o1.changeSign());
    }

    TEST_CASE ("CHANGE_SIGN_2") {
        TPNumEditor o1;
        o1.addDigit(15);
                CHECK("-F" == o1.changeSign());
    }

    TEST_CASE ("ADD_DIGIT_1") {
        TPNumEditor o1;
        o1.addDigit(5);
                CHECK("5" == o1.getNumberString());
    }

    TEST_CASE ("ADD_DIGIT_2") {
        TPNumEditor o1;
        o1.addDigit(2);
                CHECK("2" == o1.getNumberString());
    }

    TEST_CASE ("ADD_ZERO_1") {
        TPNumEditor o1;
        o1.addDigit(2);
        o1.addZero();
                CHECK("20" == o1.getNumberString());
    }

    TEST_CASE ("CLEAR_1") {
        TPNumEditor o1;
        o1.addDigit(2);
        o1.clear();
                CHECK("0" == o1.getNumberString());
    }

    TEST_CASE ("GETNUMBERSTRING_1") {
        TPNumEditor o1;
                CHECK("0" == o1.getNumberString());
    }

    TEST_CASE ("ADD_DIVIDER_1") {
        TPNumEditor o1;
        o1.addDigit(5);
        o1.addDivider();
        o1.addDigit(5);
                CHECK("5.5" == o1.getNumberString());
    }

    TEST_CASE ("SET_NUMBER_1") {
        TPNumEditor o1;
        o1.setNumber(111);
                CHECK("111" == o1.getNumberString());
    }

    TEST_CASE ("SET_NUMBER_2") {
        TPNumEditor o1;
        o1.setNumber(222);
                CHECK("222" == o1.getNumberString());
    }

    TEST_CASE ("SET_NUMBER_3") {
        TPNumEditor o1;
        o1.setNumber(333);
                CHECK("333" == o1.getNumberString());
    }

    TEST_CASE ("EDIT_NUMBER_1") {
        TPNumEditor o1;
        o1.editNumber("123");
                CHECK("123" == o1.getNumberString());
    }

    TEST_CASE ("EDIT_NUMBER_2") {
        TPNumEditor o1;
        o1.editNumber("321.321");
                CHECK("321.321" == o1.getNumberString());
    }

    TEST_CASE ("EDIT_NUMBER_3") {
        TPNumEditor o1;
        o1.editNumber("10");
                CHECK("10" == o1.getNumberString());
    }
}
