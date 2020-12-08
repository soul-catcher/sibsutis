#include "TPoly.h"


TPoly::TPoly() {}

TPoly::TPoly(int coeff, int degree) {
	polynom.emplace(degree, TMonomial(coeff, degree));
}

int TPoly::maxDegree() {
	return polynom.rbegin()->first;
}

int TPoly::coeff(int Degree) {
	if (!polynom.count(Degree))
		return 0;
	else
		return polynom.at(Degree).readCoeff();
}

void TPoly::clear() {
	polynom.clear();
}

TPoly TPoly::operator+(TPoly otherPoly) {
	TPoly result = *this;
	for (auto& pairElem : otherPoly.polynom)
		if (result.polynom.count(pairElem.first))
			result.polynom.at(pairElem.first) =
			TMonomial(result.polynom.at(pairElem.first).readCoeff() + pairElem.second.readCoeff(), pairElem.first);
		else
			result.polynom.emplace(pairElem);
	return result;
}

TPoly TPoly::operator*(TPoly otherPoly) {
	TPoly newPoly;
	for (auto& it1 : this->polynom)
		for (auto& it2 : otherPoly.polynom) {
			TMonomial newMember(it1.second.readCoeff() * it2.second.readCoeff(), it1.second.readPower() + it2.second.readPower());
			if (newPoly.polynom.count(newMember.readPower()))
				newPoly.polynom.emplace(newMember.readPower(), TMonomial(newMember.readCoeff() + newPoly.polynom.at(newMember.readPower()).readCoeff(), newMember.readPower()));
			else
				newPoly.polynom.emplace(newMember.readPower(), newMember);
		}
	return newPoly;
}

TPoly TPoly::operator-(TPoly otherPoly) {
	TPoly result = *this;
	for (auto& pairElem : otherPoly.polynom)
		if (result.polynom.count(pairElem.first))
			result.polynom.at(pairElem.first) =
			TMonomial(result.polynom.at(pairElem.first).readCoeff() - pairElem.second.readCoeff(), pairElem.first);
		else
			result.polynom.emplace(-pairElem.first, TMonomial(-pairElem.first, pairElem.second.readPower()));
	return result;
}

TPoly TPoly::minus() {
	TPoly newPoly;
	for (auto& it : polynom)
		newPoly.polynom.emplace(-it.first, TMonomial(-it.second.readCoeff(), it.second.readPower()));
	return newPoly;
}


TPoly TPoly::differentiate() {
	TPoly newPoly;
	for (auto& it : polynom)
		newPoly.polynom.emplace(it.first == 0 ? 0 : it.first - 1, it.second.differentiate());
	return newPoly;
}

double TPoly::compute(double x) {
	if (x == 0)
		return 0;
	double sum = 0.0;
	for (auto& it : polynom)
		sum += it.second.compute(x);
	if (sum > std::numeric_limits<double>::max()) {
		throw overflow_error("Overflow");

	}
	return sum;
}

TMonomial TPoly::elem(int pos) {
	if (pos < 0 || pos >(int)polynom.size())
		return TMonomial();
	else {
		int cntr = 0;
		for (auto& it : polynom)
			if (cntr == pos)
				return it.second;
	}
	return TMonomial();
}

TMonomial TPoly::operator[](int pos) {
	if (pos < 0 || pos > (int)polynom.size())
		return TMonomial();
	else {
		int cntr = 0;
		for (auto& it : polynom)
			if (cntr++ == pos)
				return it.second;
	}
	return TMonomial();
}

TPoly::~TPoly() {}