#!/usr/bin/env python3

import sys

from lib import *

text = prepare_text(open(sys.argv[1]).read())

for i in range(1, 4):
    subs = list(gen_subsequences(text, i))
    freqs = get_frequencies(subs)
    estimation = entropy(freqs.values()) / i
    print(
        f'\n{i = }',
        f'{estimation = }',
        sep='\n',
    )
