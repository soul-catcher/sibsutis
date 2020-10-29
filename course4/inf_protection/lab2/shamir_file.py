from crypto import *

p = gen_p(1 << 8, 10 ** 9)


def gen_c_d(p):
    while True:
        c = gen_mutually_prime(p - 1)
        gcd, d, _ = extgcd(c, p - 1)
        assert gcd == 1
        if d > 0:
            break
    return c, d


ca, da = gen_c_d(p)
cb, db = gen_c_d(p)

open('keys.txt', 'w').write(f'p = {p}\nca = {ca}\tda = {da}\ncb = {cb}\tdb = {db}')

original_data = open('original.jpg', 'rb').read()


def encrypt_data(data, power, mod):
    return [pow(byte, power, mod) for byte in data]


alice_encrypted = encrypt_data(original_data, ca, p)  # Зашифровка Алисой
alice_and_bob_encrypted = encrypt_data(alice_encrypted, cb, p)  # Зашифровка Бобом
open('shamir_encrypted.txt', 'w').write(str(alice_and_bob_encrypted))
bob_encrypted = encrypt_data(alice_and_bob_encrypted, da, p)  # Расшифровка Алисой
decrypted = encrypt_data(bob_encrypted, db, p)  # Расшифровка Бобом

write_bytes_to_file(decrypted, open('shamir_decrypted.jpg', 'wb'))
