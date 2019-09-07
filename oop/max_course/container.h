#pragma once

class Container {
public:

    virtual void clear() = 0;

    virtual bool empty() const = 0;
};

template<typename T>
class SequenceContainer : public Container {
protected:
    T *arr;
public:
    SequenceContainer(T *arr = nullptr) : arr(arr) {};

    void clear() override {
        arr = nullptr;
    }

    bool empty() const override {
        return !arr;
    }

    T operator[](int i) {
        return arr[i];
    };
};
