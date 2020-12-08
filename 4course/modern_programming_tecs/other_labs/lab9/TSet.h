#pragma once
#include <set>
#include <algorithm>
using namespace std;

template <class T>

class TSet {

public:
	set<T> container;
	TSet();
	void clear();
	void insert_(T a);
	void del(T a);
	bool isEmpty();
	bool contains(T a);
	TSet<T> add(const TSet<T>& otherSet);
	TSet<T> subtract(TSet<T> set);
	TSet<T> multiply(TSet<T> set);
	int count();
	T element(int num);
	~TSet();
};

template<class T>
TSet<T>::TSet() {}

template<class T>
void TSet<T>::clear() {
	container.clear();
}

template<class T>
void TSet<T>::insert_(T a) {
	container.insert(a);
}

template<class T>
void TSet<T>::del(T a) {
	if (container.find(a) != container.end())
		container.erase(a);
}

template<class T>
bool TSet<T>::isEmpty() {
	return container.empty();
}

template<class T>
bool TSet<T>::contains(T a) {
	return container.find(a) != container.end();
}

template<class T>
TSet<T> TSet<T>::add(const TSet<T>& otherSet) {
	TSet<T> result = *this;
	for (const T& a : otherSet.container)
		result.insert_(a);
	return result;
}

template<class T>
TSet<T> TSet<T>::subtract(TSet<T> otherSet) {
	TSet<T> result = *this;
	for (const T& a : otherSet.container)
		if (result.container.find(a) != result.container.end())
			result.del(a);
	return result;
}

template<class T>
TSet<T> TSet<T>::multiply(TSet<T> otherSet) {
	TSet<T> result;
	for (const T& a : otherSet.container)
		if (container.count(a))
			result.insert_(a);
	return result;
}

template<class T>
int TSet<T>::count() {
	return container.size();
}

template<class T>
T TSet<T>::element(int num) {
	if (num >=0 && num < container.size())
		return *next(container.begin(), num);
	throw invalid_argument("Invalid type");
	T abc;
	return abc;
}

template<class T>
TSet<T>::~TSet() {}