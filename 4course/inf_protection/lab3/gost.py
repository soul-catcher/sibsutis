from hashlib import md5

from crypto import *

q = gen_p(1 << 255, (1 << 256) - 1)

while True:  # Генерируем p
    b = random.randint(math.ceil((1 << 1023) / q), ((1 << 1024) - 1) // q)
    if is_prime(p := b * q + 1):
        break

while True:  # Находим a
    g = random.randrange(2, p - 1)
    if (a := pow(g, b, p)) > 1:
        break

x = random.randrange(1, q)  # Закрытый ключ
y = pow(a, x, p)  # Открытый ключ

h = int.from_bytes(md5(open('../res/original.jpg', 'rb').read()).digest(), byteorder=sys.byteorder)
assert 0 < h < q
while True:
    k = random.randrange(1, q)
    if (r := pow(a, k, p) % q) == 0:
        continue
    if (s := (k * h % q + x * r % q) % q) != 0:
        break
open('generated/gost.txt', 'w').write(f'{r = }\n{s = }')

# Проверка подписи
assert 0 < r < q
assert 0 < s < q
gcd, hh, _ = extgcd(h, q)
u1 = s * hh % q
u2 = -r * hh % q
v = pow(a, u1, p) * pow(y, u2, p) % p % q
assert v == r
print(f'{q = }\n{p = }\n{a = }\n{x = }\n{y = }\n{h = }\n{k = }\n{s = }\n{u1 = }\n{u2 = }\n{r = }\n{v = }')