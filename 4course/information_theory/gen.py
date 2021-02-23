#!/usr/bin/env python3

import argparse
import random

parser = argparse.ArgumentParser(
    description="Generate a k elements chosen from the population with replacement. "
    "If a weights sequence is specified, selections are made according to the relative weights. "
    "Weights sequence must be the same length as the population sequence."
)
parser.add_argument("-p", "--population", required=True, help="Population of elements")
parser.add_argument("-w", "--weights", type=float, nargs="+", help="Population weights")
parser.add_argument(
    "-k", metavar="K", required=True, type=int, help="Amount of generated characters"
)

args = parser.parse_args()

print(*random.choices(**vars(args)), sep="")
