#!/usr/bin/env python3

import sys

from lib import *

text = open(sys.argv[1]).read()

print(
    f'Alphabet = {repr("".join(set(text)))}',
    f'Alphabet size = {len(set(text))}',
    sep='\n'
)

for i in range(1, 4):
    subs = list(gen_subsequences(text, i))
    freqs = get_frequencies(subs)
    print(
        f'\n{i = }',
        f'Entropy = {entropy(freqs.values()) / i}',
        sep='\n'
    )
