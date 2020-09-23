"""Криптографическая библиотека"""

__author__ = "Dmitry Samsonov"

import math
import random
import typing


def fast_pow(base: int, power: int, mod: int = None) -> int:
    """
    Быстрое возведение числа в степень по модулю.

    Реализация алгоритмма имеет трудоёмкость O(log2(power)).
    Показатель не может быть отрицательным, модуль не может быть равным нулю.
    Если модуль не был дан, проводит простое быстрое возведения в степень.

    Пример:

    fast_pow(5, 2206, 22) -> 5
    """
    if mod == 0:
        raise ValueError("Модуль не может быть равен нулю")
    if power < 0:
        raise ValueError("Показатель не может быть отрицательным")
    result = 1
    if mod:
        base %= mod
        result %= mod  # Для правильного вычисления, когда показатель равен 0, а модуль отрицателен
    while power:
        power, remainder = divmod(power, 2)
        if remainder:
            result *= base
            if mod:
                result %= mod
        base *= base
        if mod:
            base %= mod
    return result


def extgcd(a: int, b: int) -> typing.Tuple[int, int, int]:
    """
    Обобщённый алгоритм Евклида.

    Возвращает наибольший общий делитель и их коэффициенты Безу.
    Оба числа должны быть натуральными.

    Пример:

    egcd(12, 8) -> (4, 1, -1), при том 4 = 1 * 12 - 1 * 8
    """
    if a <= 0 or b <= 0:
        raise ValueError("Числа могут быть только натуральными")
    # if a < b:
    #     a, b = b, a  # Ломает алгоритм
    u1, u2, u3 = a, 1, 0
    v1, v2, v3 = b, 0, 1
    while v1:
        q = u1 // v1
        t1, t2, t3 = u1 % v1, u2 - q * v2, u3 - q * v3
        u1, u2, u3 = v1, v2, v3
        v1, v2, v3 = t1, t2, t3
    return u1, u2, u3


def is_prime(n: int) -> bool:
    """Возвращает True, если число n простое, иначе False"""
    if n < 2:
        return False
    return all(n % i for i in range(2, int(math.sqrt(n)) + 1))


def gen_p(a: int, b: int) -> int:
    """Генерирует безопасное простое число в диапазоне [a, b]"""
    while True:
        num = random.randint(a // 2, (b - 1) // 2)
        if is_prime(num) and is_prime(num * 2 + 1):
            return num * 2 + 1


def _gen_g(mod: int) -> int:
    return next(g for g in range(2, mod - 1) if pow(g, (mod - 1) // 2, mod) != 1)


def gen_public(private_key: int, mod: int):
    """Генерирует открытый ключ из закрытого по модулю"""
    return pow(_gen_g(mod), private_key, mod)


def gen_common(secret_key: int, public_key: int, mod: int) -> int:
    """Генерирует общий ключ из закрытого и открытого по модулю"""
    return pow(public_key, secret_key, mod)
