#pragma once

enum TOperations {
    None, Add, Sub, Mul, Div
};

enum TFunc {
    Rev, Sqr
};

template<typename T = int>
class TProc {
private:
    T Lop_Res;
    T Rop;
    TOperations operation;
public:
    TProc(const T &leftObj, const T &rightObj);

    void resetProc();

    void resetOper();

    void doOper();

    void doFunc(TFunc func);

    T readLeft();

    T readRight();

    void writeLeft(const T &someObj);

    void writeRight(const T &someObj);

    void writeOper(TOperations someOper);

    TOperations readOper();

    ~TProc();
};

template<typename T>
TProc<T>::TProc(const T &leftObj, const T &rightObj) {
    operation = None;
    this->Lop_Res = leftObj;
    this->Rop = rightObj;
}

template<typename T>
void TProc<T>::resetProc() {
    operation = None;
    T clearObj;
    this->Lop_Res = this->Rop = clearObj;
}

template<typename T>
void TProc<T>::resetOper() {
    operation = None;
}

template<typename T>
void TProc<T>::doOper() {
    switch (operation) {
        case Add:
            Lop_Res = Lop_Res + Rop;
            break;
        case Sub:
            Lop_Res = Lop_Res - Rop;
            break;
        case Mul:
            Lop_Res = Lop_Res * Rop;
            break;
        case Div:
            Lop_Res = Lop_Res / Rop;
            break;
        default:
            break;
    }
}

template<typename T>
void TProc<T>::doFunc(TFunc func) {
    switch (func) {
        case Rev:
            Rop = Rop.inverse();
            break;
        case Sqr:
            Rop = Rop * Rop;
            break;
        default:
            break;
    }
}

template<typename T>
T TProc<T>::readLeft() {
    return T(this->Lop_Res);
}

template<typename T>
T TProc<T>::readRight() {
    return T(this->Rop);
}

template<typename T>
void TProc<T>::writeLeft(const T &someObj) {
    this->Lop_Res = someObj;
}

template<typename T>
void TProc<T>::writeRight(const T &someObj) {
    this->Rop = someObj;
}

template<typename T>
void TProc<T>::writeOper(TOperations someOper) {
    this->operation = someOper;
}

template<typename T>
TOperations TProc<T>::readOper() {
    return operation;
}

template<typename T>
TProc<T>::~TProc() {}