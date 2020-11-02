import itertools
import operator

from crypto import *

original_data = open('original.jpg', 'rb').read()
key = [random.randrange(1 << 8) for _ in range(len(original_data))]
encrypted = list(itertools.starmap(operator.xor, zip(original_data, key)))
open('generated/vernam_encrypted.txt', 'w').write(str(encrypted))
decrypted = list(itertools.starmap(operator.xor, zip(encrypted, key)))
write_bytes_to_file(decrypted, open('generated/vernam_decrypted.jpg', 'wb'))
