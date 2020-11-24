#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../doctest.h"

#include "../lab1/TFrac.h"
#include "TProc.h"

TEST_SUITE ("TProc_Test") {
    TEST_CASE ("Init_1") { // 0/1 0/1
        TFrac leftFrac;
        TFrac rightFrac;
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFrac answer;
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Init_2") { // 11/3 11/3
        TFrac leftFrac(11, 3);
        TFrac rightFrac;
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFrac answer(11, 3);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Init_3") { // 17/9 17/9
        TFrac leftFrac(16, 4);
        TFrac rightFrac(17, 9);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFrac answer(17, 9);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Init_4") { // 17/9 17/9
        TFrac leftFrac(500, 4);
        TFrac rightFrac(-300, 9);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFrac answer(-300, 9);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Init_5") { // 17/9 17/9
        TFrac leftFrac(500, 4);
        TFrac rightFrac(123, 45);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFrac answer(123, 45);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Operation_1") { // 1/2 + 1/2 = 1/1
        TFrac leftFrac(1, 2);
        TFrac rightFrac(1, 2);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Add;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(1, 1);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_2") { // 3/4 - 5/6 = -1/12
        TFrac leftFrac(3, 4);
        TFrac rightFrac(5, 6);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Sub;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(-1, 12);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_3") { // 12/7 * 5/9 = 20/21
        TFrac leftFrac(12, 7);
        TFrac rightFrac(5, 9);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Mul;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(20, 21);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_4") { // 56/7 : -22/3 = -12/11
        TFrac leftFrac(56, 7);
        TFrac rightFrac(-22, 3);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Div;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(-12, 11);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_5") {
        TFrac leftFrac(15, 10);
        TFrac rightFrac(55, 60);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Add;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(29, 12);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_6") {
        TFrac leftFrac(1, 1);
        TFrac rightFrac(1, 1);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Add;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(2, 1);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_7") {
        TFrac leftFrac(3, 1);
        TFrac rightFrac(1, 1);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Add;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(4, 1);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_8") { // 12/7 * 5/9 = 20/21
        TFrac leftFrac(7, 7);
        TFrac rightFrac(7, 7);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Mul;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(1, 1);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_9") { // 12/7 * 5/9 = 20/21
        TFrac leftFrac(8, 7);
        TFrac rightFrac(8, 7);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Mul;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(64, 49);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Operation_10") { // 12/7 * 5/9 = 20/21
        TFrac leftFrac(5, 7);
        TFrac rightFrac(1, 1);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TOperations oper = Mul;
        obj.writeOper(oper);
        obj.doOper();
        TFrac answer(5, 7);
                CHECK(answer.getFractionString() == obj.readLeft().getFractionString());
    }

    TEST_CASE ("Function_1") { // reverse
        TFrac leftFrac(56, 7);
        TFrac rightFrac(-22, 3);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFunc funcOper = Rev;
        obj.doFunc(funcOper);
        TFrac answer(-3, 22);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Function_2") {
        TFrac leftFrac;
        TFrac rightFrac(-22, 3);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFunc funcOper = Sqr;
        obj.doFunc(funcOper);
        TFrac answer(484, 9);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Function_3") {
        TFrac leftFrac;
        TFrac rightFrac(1, 1);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFunc funcOper = Sqr;
        obj.doFunc(funcOper);
        TFrac answer(1, 1);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Function_4") {
        TFrac leftFrac;
        TFrac rightFrac(2, 2);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFunc funcOper = Sqr;
        obj.doFunc(funcOper);
        TFrac answer(1, 1);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

    TEST_CASE ("Function_5") {
        TFrac leftFrac;
        TFrac rightFrac(3, 9);
        TProc<TFrac> obj(leftFrac, rightFrac);
        TFunc funcOper = Sqr;
        obj.doFunc(funcOper);
        TFrac answer(1, 9);
                CHECK(answer.getFractionString() == obj.readRight().getFractionString());
    }

}
