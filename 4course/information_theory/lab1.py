#!/usr/bin/env python3
import math
from itertools import product

from lib import *

PROBABILITIES = [0.1, 0.7, 0.1, 0.1]

f1 = open('f1.txt').readline().removesuffix('\n')
f2 = open('f2.txt').readline().removesuffix('\n')

for i in range(1, 4):
    f1_subs = list(gen_subsequences(f1, i))
    f2_subs = list(gen_subsequences(f2, i))
    f1_freqs = get_frequencies(f1_subs)
    f2_freqs = get_frequencies(f2_subs)
    f1_estimation = entropy(f1_freqs.values()) / i
    f2_estimation = entropy(f2_freqs.values()) / i
    f2_theoretical_probs = list(map(math.prod, product(PROBABILITIES, repeat=i)))
    f1_theoretical = math.log2(len(f2_theoretical_probs)) / i
    f2_theoretical = entropy(f2_theoretical_probs) / i
    print(
        f'\n{i = }',
        f'{f1_estimation = }',
        f'{f2_estimation = }',
        f'{f1_theoretical = }',
        f'{f2_theoretical = }',
        sep='\n',
    )
