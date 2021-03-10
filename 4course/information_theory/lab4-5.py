#!/usr/bin/env python3

from lib import *

f1 = open('f1.txt').readline().removesuffix('\n')
f2 = open('f2.txt').readline().removesuffix('\n')
dog_hearth = prepare_text(open('dog-hearth.txt').read())
code = open('lib.py').read()

for i, text in enumerate((f1, f2, dog_hearth, code), 1):
    print(f' text {i} '.center(80, '='))
    frequencies = get_frequencies(text)
    text_entropy = entropy(frequencies.values())
    for n in range(2, 5):
        print(f' encoded alphabet len = {n} '.center(40, '*'))
        code_table = huffman(frequencies, n)
        if n == 2:
            redundancy = sum(map(len, code_table.values())) / len(code_table) - text_entropy
            print(f'{redundancy = }')

        encoded_text = ''.join(map(str, encode(text, code_table)))
        for sub_len in range(1, 4):
            subs = list(gen_subsequences(encoded_text, sub_len))
            sub_freqs = get_frequencies(subs)
            print(
                f'{sub_len = }',
                f'Entropy = {entropy(sub_freqs.values()) / sub_len}',
                sep='\n'
            )
