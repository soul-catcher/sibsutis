import sys

from crypto import *

MESSAGE_SIZE = 1  # Размер сообщения в байтах

p = gen_p(1 << MESSAGE_SIZE * 8, 10 ** 9)

while True:
    ca = gen_mutually_prime(p - 1)
    gcd, da, _ = extgcd(ca, p - 1)
    assert gcd == 1
    if da > 0:
        break

while True:
    cb = gen_mutually_prime(p - 1)
    gcd, db, _ = extgcd(cb, p - 1)
    assert gcd == 1
    if db > 0:
        break

original_f = open('original.jpg', 'rb')
alice_encrypted_f = open('alice_encrypted.bin', 'wb+')
alice_and_bob_encrypted_f = open('alice_and_bob_encrypted.bin', 'wb+')
bob_encrypted_f = open('bob_encrypted.bin', 'wb+')
decrypted_f = open('decrypted.jpg', 'wb')


def encrypt_file(original_file, encrypted_file, message_size, power, mod):
    while byte := original_file.read(message_size):
        encrypted_file.write(pow(int.from_bytes(byte, sys.byteorder), power, mod).to_bytes(MESSAGE_SIZE, sys.byteorder))


# Зашифровка файла Алисой
encrypt_file(original_f, alice_encrypted_f, MESSAGE_SIZE, ca, p)
original_f.close()
alice_encrypted_f.seek(0)

# Зашифровка файла Бобом
encrypt_file(alice_encrypted_f, alice_and_bob_encrypted_f, MESSAGE_SIZE, cb, p)
alice_encrypted_f.close()
alice_and_bob_encrypted_f.seek(0)

# Расшифровка файла Алисой
encrypt_file(alice_and_bob_encrypted_f, bob_encrypted_f, MESSAGE_SIZE, da, p)
alice_and_bob_encrypted_f.close()
bob_encrypted_f.seek(0)

# Расшифровка файла Бобом
encrypt_file(bob_encrypted_f, decrypted_f, MESSAGE_SIZE, db, p)
alice_and_bob_encrypted_f.close()
