from hashlib import md5

from crypto import *

h = md5(open('../res/original.jpg', 'rb').read())
h_int = int.from_bytes(h.digest(), byteorder=sys.byteorder)  # Хеш файла

p = gen_safe_p(1 << h.digest_size * 8 + 1, 1 << h.digest_size * 8 + 2)

g = gen_g(p)
x = random.randrange(2, p - 1)  # Закрытый ключ
y = pow(g, x, p)  # Открытый ключ
k = gen_mutually_prime(p - 1)
r = pow(g, k, p)
u = (h_int - x * r) % (p - 1)
gcd, kk, _ = extgcd(k, p - 1)
assert gcd == 1
s = kk * u % (p - 1)
open('generated/elgamal.txt', 'w').write(str(s))

# Проверка подписи
yr = pow(y, r, p) * pow(r, s, p) % p
gh = pow(g, h_int, p)
assert yr == gh
print(f'{h_int = }\n{p = }\n{g = }\n{x = }\n{y = }\n{k = }\n{r = }\n{u = }\n{s = }\n{yr = }\n{gh = }')
