from crypto import *

p = gen_p(1 << 8, 10 ** 9)

ca, da = gen_c_d(p - 1)
cb, db = gen_c_d(p - 1)

open('generated/shamir_keys.txt', 'w').write(f'p = {p}\nca = {ca}\tda = {da}\ncb = {cb}\tdb = {db}')

original_data = open('../res/original.jpg', 'rb').read()


def encrypt_data(data, power, mod):
    return [pow(byte, power, mod) for byte in data]


alice_encrypted = encrypt_data(original_data, ca, p)  # Зашифровка Алисой
alice_and_bob_encrypted = encrypt_data(alice_encrypted, cb, p)  # Зашифровка Бобом
open('generated/shamir_encrypted.txt', 'w').write(str(alice_and_bob_encrypted))
bob_encrypted = encrypt_data(alice_and_bob_encrypted, da, p)  # Расшифровка Алисой
decrypted = encrypt_data(bob_encrypted, db, p)  # Расшифровка Бобом

write_bytes_to_file(decrypted, open('generated/shamir_decrypted.jpg', 'wb'))
