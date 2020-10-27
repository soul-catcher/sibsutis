from crypto import *
# p = 23
# message = 12
p = gen_p(1, 10 ** 9)
message = random.randint(1, p)

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

x1 = pow(message, ca, p)
x2 = pow(x1, cb, p)
x3 = pow(x2, da, p)
x4 = pow(x3, db, p)
print(f'p = {p}, message = {message}\nca = {ca}, da = {da}\ncb = {cb}, db = {db}\n'
      f'x1 = {x1}\nx2 = {x2}\nx3 = {x3}\nx4 = {x4}')
assert message == x4
