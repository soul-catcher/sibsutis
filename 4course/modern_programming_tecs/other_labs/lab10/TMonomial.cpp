#include "TMonomial.h"
#include <cmath>
#include <cfloat>


TMonomial::TMonomial() {}

TMonomial::TMonomial(int coeff, int power) {
	this->coeff = coeff;
	if (coeff == 0)
	{
		this->power = 0;
	}
	else
	{
		this->power = power;
	}
}

int TMonomial::readPower() {
	return this->power;
}

void TMonomial::writePower(int power) {
	this->power = power;
}

int TMonomial::readCoeff() {
	return this->coeff;
}

void TMonomial::writeCoeff(int coeff) {
	this->coeff = coeff;
}

bool TMonomial::isEqual(TMonomial c) {
	return (this->coeff == c.coeff && this->power == c.power);
}

TMonomial TMonomial::differentiate() {
	return TMonomial(this->power == 0 ? 0 : this->coeff * this->power,
		this->power == 0 ? 0 : this->power - 1);
}

double TMonomial::compute(double x) {
	if ((this->coeff * pow(x, this->power)) > DBL_MAX) {
		throw overflow_error("Overflow");
		return (-99999);
	}
	return (this->coeff * pow(x, this->power));
}

string TMonomial::monomialToString() {
	if (this->coeff == 0)
		return string("0");
	else
		return string(to_string(this->coeff)
			+ "*x^" + to_string(this->power));
}

TMonomial::~TMonomial() {}