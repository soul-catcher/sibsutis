from crypto import *

bob_private = 228
alice_private = 1488

mod = gen_p(2000, 3000)
bob_public = gen_public(bob_private, mod)
alice_public = gen_public(alice_private, mod)
bob_common = gen_common(bob_private, alice_public, mod)
alice_common = gen_common(alice_private, bob_public, mod)
print(bob_common)
assert bob_common == alice_common
