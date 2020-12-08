#pragma once
#include "TMonomial.h"
#include <map>
class TPoly {
private:
	map<int, TMonomial> polynom;
public:
	TPoly();
	TPoly(int coeff, int degree);
	int maxDegree();
	int coeff(int Degree);
	void clear();
	TPoly operator+ (TPoly otherPoly);
	TPoly operator- (TPoly otherPoly);
	TPoly operator* (TPoly otherPoly);
	TPoly minus();
	bool operator == (const TPoly& otherPoly) const {
		return this->polynom == otherPoly.polynom;
	}
	TPoly differentiate();
	double compute(double x);
	TMonomial elem(int pos);
	TMonomial operator[] (int pos);
	~TPoly();
};