from hashlib import md5

from crypto import *

h = md5(open('../res/original.jpg', 'rb').read())
h_int = int.from_bytes(h.digest(), byteorder=sys.byteorder)  # Хеш файла

p = gen_p(1 << h.digest_size * 8 // 2 + 1, 1 << h.digest_size * 8 // 2 + 2)
q = gen_p(1 << h.digest_size * 8 // 2 + 1, 1 << h.digest_size * 8 // 2 + 2)
assert p != q
n = p * q
phi = (p - 1) * (q - 1)
d, c = gen_c_d(phi)  # Открытый и закрытый ключ

s = pow(h_int, c, n)  # Подпись
open('generated/rsa.txt', 'w').write(str(s))
# Проверка подписи
e = pow(s, d, n)
assert e == h_int
print(f'{p = }\n{q = }\n{n = }\n{phi = }\n{d = }\n{c = }\n{h_int = }\n{s = }\n{e = }')
