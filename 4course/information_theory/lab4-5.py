#!/usr/bin/env python3

from pprint import pprint

from lib import *

f1 = open('f1.txt').readline().removesuffix('\n')
f2 = open('f2.txt').readline().removesuffix('\n')
dog_hearth = prepare_text(open('dog-hearth.txt').read())
code = open('lib.py').read()

for text in f1, f2, dog_hearth, code:
    for n in range(2, 5):
        code_table = huffman(get_frequencies(text), n)
        encoded_text = encode(text, code_table)

        # for i in range(1, 4):
        #     subsequences = [list(gen_subsequences(list(code.values()), i)) for code in codes]
        #     entropys = list(map(entropy, (get_frequencies(list(subsequence)).values() for subsequence in subsequences)))
        #     pprint(entropys)
