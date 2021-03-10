#!/usr/bin/env python3

import sys
from pprint import pprint

from lib import *

# text = open(sys.argv[1]).read()

text = 'fffh'
freqs = get_frequencies(text)

pprint(huffman(freqs, 3))