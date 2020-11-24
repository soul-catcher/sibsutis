import unittest

from crypto import *


class TestFastPow(unittest.TestCase):
    def test_simple_pow(self):
        params = 16, 21
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_base_is_zero(self):
        params = 0, 21, 5
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_power_is_zero(self):
        params = 7, 0, 5
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_mod_is_zero(self):
        params = 3, 3, 0
        with self.assertRaises(ValueError):
            fast_pow(*params)

    def test_base_and_power_are_zeroes(self):
        params = 0, 0, 5
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_base_is_negative(self):
        params = -16, 21, 5
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_power_is_negative(self):
        params = 7, -16, 5
        with self.assertRaises(ValueError):
            fast_pow(*params)

    def test_mod_is_negative(self):
        params = 16, 33, -11
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_base_and_mod_are_negatives(self):
        params = -12, 33, -11
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_small_positive_params(self):
        for a in range(3):
            for power in range(3):
                for mod in range(1, 3):
                    with self.subTest(f"a = {a}, power = {power}, mod = {mod}"):
                        self.assertEqual(pow(a, power, mod), fast_pow(a, power, mod))

    def test_small_negative_base_and_mod_but_positive_power(self):
        for a in range(0, -3, -1):
            for power in range(3):
                for mod in range(-1, -3, -1):
                    with self.subTest(f"a = {a}, power = {power}, mod = {mod}"):
                        self.assertEqual(pow(a, power, mod), fast_pow(a, power, mod))

    def test_huge_base(self):
        params = 172357347057984687152457294381, 17, 23
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_huge_pow(self):
        params = 13, 4536472307498614467129865, 77
        self.assertEqual(pow(*params), fast_pow(*params))

    def test_huge_mod(self):
        params = 13, 22, 4735468250429017661326498
        self.assertEqual(pow(*params), fast_pow(*params))


class TestEgcd(unittest.TestCase):
    def test_ones(self):
        self.assertEqual((1, 0, 1), extgcd(1, 1))

    def test_simple(self):
        self.assertEqual((4, 1, -1), extgcd(12, 8))

    def test_simple_another_order(self):
        self.assertEqual((4, -1, 1), extgcd(8, 12))

    def test_equals(self):
        self.assertEqual((7, 0, 1), extgcd(7, 7))

    def test_big_nums(self):
        self.assertEqual((2, 2437250447493, -2431817869532), extgcd(23894798501898, 23948178468116))

    def test_powers_of_2(self):
        self.assertEqual((1, -260414429242905345185687, 408415383037561), extgcd(pow(2, 50), pow(3, 50)))

    def test_zeroes(self):
        with self.assertRaises(ValueError):
            extgcd(0, 0)

    def test_negatives(self):
        with self.assertRaises(ValueError):
            extgcd(-10, -10)


class TestIsPrime(unittest.TestCase):
    def test_zero(self):
        self.assertFalse(is_prime(0))

    def test_one(self):
        self.assertFalse(is_prime(1))

    def test_some_primes(self):
        for prime in 2, 3, 5, 7, 11, 13:
            with self.subTest(prime):
                self.assertTrue(is_prime(prime))

    def test_some_composites(self):
        for composite in 4, 6, 8, 9, 10:
            with self.subTest(composite):
                self.assertFalse(is_prime(composite))


class TestShanks(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(2, shanks(1, 2, 3))

    def test_simple_2(self):
        self.assertEqual(3, shanks(2, 3, 5))

    def test_none(self):
        self.assertIsNone(shanks(4, 34, 7))

    def test_mod_less_y(self):
        with self.assertRaises(ValueError):
            shanks(30, 5, 18)


if __name__ == '__main__':
    unittest.main()
