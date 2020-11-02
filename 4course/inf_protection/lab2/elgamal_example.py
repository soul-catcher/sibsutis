from crypto import *
p = gen_safe_p(1, 10 ** 9)
g = gen_g(p)
cb = random.randrange(1, p)
m = random.randrange(1, p)  # message
k = random.randint(1, p - 2)
db = pow(g, cb, p)
r = pow(g, k, p)
e = m * pow(db, k, p) % p
mm = e * pow(r, p - 1 - cb, p) % p
print(f'm = {m}\np = {p}\ng = {g}\ncb = {cb}\tdb = {db}\nk = {k}\tr = {r}\nmm = {mm}')