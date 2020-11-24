from crypto import *

pb, qb = gen_p(1 << 8, 10 ** 9), gen_p(1 << 8, 10 ** 9)
nb = pb * qb
phi = (pb - 1) * (qb - 1)
db, cb = gen_c_d(phi)

open('generated/rsa_keys.txt', 'w').write(f'pb = {pb}\tqb = {qb}\tnb = {nb}\nphi = {phi}\ndb = {db}\tcb = {cb}')

original_data = open('../res/original.jpg', 'rb').read()
encrypted = [pow(byte, db, nb) for byte in original_data]
open('generated/rsa_encrypted.txt', 'w').write(str(encrypted))
decrypted = [pow(byte, cb, nb) for byte in encrypted]
write_bytes_to_file(decrypted, open('generated/rsa_decrypted.jpg', 'wb'))
