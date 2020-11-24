from crypto import *
p = gen_safe_p(1 << 8, 10 ** 9)
g = gen_g(p)
cb = random.randrange(1, p)  # private
db = pow(g, cb, p)  # public

k = random.randint(1, p - 2)
r = pow(g, k, p)

open('generated/elgamal_keys.txt', 'w').write(f'p = {p}\ng = {g}\ncb = {cb}\tdb = {db}\nk = {k}\tr = {r}')

original_data = open('../res/original.jpg', 'rb').read()

encrypted = [byte * pow(db, k, p) % p for byte in original_data]
open('generated/elgamal_encrypted.txt', 'w').write(str(encrypted))
decrypted = [byte * pow(r, p - 1 - cb, p) % p for byte in encrypted]
write_bytes_to_file(decrypted, open('generated/elgamal_decrypted.jpg', 'wb'))
