#pragma once
#include <string>
#include <stdexcept>
using namespace std;

class TMonomial {
private:
	int coeff;
	int power;
public:
	TMonomial();
	TMonomial(int coeff, int power);
	int readPower();
	void writePower(int power);
	int readCoeff();
	void writeCoeff(int coeff);
	bool isEqual(TMonomial comparable);
	TMonomial differentiate();
	double compute(double x);
	string monomialToString();
	bool operator == (const TMonomial& comparable) const {
		return this->power == comparable.power && this->coeff == comparable.coeff;
	}
	~TMonomial();
};