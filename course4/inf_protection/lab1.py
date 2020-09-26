#!/usr/bin/env python3
from crypto import *

p = gen_p(1, 10 ** 9)
a = random.randint(1, 10 ** 9)
y = random.randint(1, p)
x = shanks(y, a, p)
power = fast_pow(a, x, p)
print(f"a = {a}, p = {p}, y = {y}, x = {x}, power = {power}")
