from crypto import *

pb, qb = gen_p(1 << 8, 10 ** 9), gen_p(1 << 8, 10 ** 9)
nb = pb * qb
phi = (pb - 1) * (qb - 1)
m = random.randrange(1, nb)  # message

db, cb = gen_c_d(phi)
e = pow(m, db, nb)
mm = pow(e, cb, nb)
print(f'm = {m}\npb = {pb}\tqb = {qb}\tnb = {nb}\nphi = {phi}\ndb = {db}\tcb = {cb}\te = {e}\nmm = {mm}')