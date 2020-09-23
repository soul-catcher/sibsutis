#!/usr/bin/env python3

import argparse
import json
import sys
import typing
from dataclasses import dataclass

from graphical_tree import GraphicalTree, Vertex


def check_positive(value: str) -> int:
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value} не является натуральным числом") from e
    return ivalue


parser = argparse.ArgumentParser()
parser.add_argument(metavar="FILE", type=argparse.FileType(), help="JSON файл, содержащий грамматику",
                    dest="grammar_file")
parser.add_argument("-r", "--range", type=check_positive, required=True,
                    help="2 натуральных числа - диапазон длин генерируемых цепочек", nargs=2, metavar=("FROM", "TO"))
args = parser.parse_args()
if args.range[0] > args.range[1]:
    args.range[0], args.range[1] = args.range[1], args.range[0]


@dataclass(frozen=True)
class Grammar:
    VT: typing.List[str]
    VN: typing.List[str]
    P: typing.Dict[str, typing.List[str]]
    S: str


try:
    G = Grammar(**json.load(args.grammar_file))
except (json.JSONDecodeError, TypeError):
    print("Файл грамматики некоректен", file=sys.stderr)
    exit(1)

stack = [([], G.S)]
was_in_stack = set()
counter = 1
ans = []
try:
    while stack:
        prev, sequence = stack.pop()
        prev = prev.copy()
        prev.append(sequence)
        if sequence in was_in_stack:
            continue
        was_in_stack.add(sequence)
        only_term = True
        for i, symbol in enumerate(sequence):
            if symbol in G.VN:
                only_term = False
                for elem in G.P[symbol]:
                    scopy = sequence[:i] + elem + sequence[i + 1:]
                    if len(scopy) <= args.range[1] + 1:
                        stack.append((prev, scopy))
        if only_term and args.range[0] <= len(sequence) <= args.range[1]:
            ans.append(prev)
            print(f"[{counter}]", sequence if sequence else "λ")
            counter += 1
except KeyError:
    print("Ошибка. Вероятнее всего грамматика задана некорректно!", file=sys.stderr)
    exit(1)

while True:
    choise = input("Какую цепочку построить?\n> ")
    if choise.isdecimal() and 1 <= int(choise) <= len(ans):
        break
    print("Введите натуральное число, не превышающее количества цепочек")
choised_ans = ans[int(choise) - 1]


def get_changes(current, next):
    if len(next) < len(current):
        return "λ"
    for i, ch in enumerate(current[::-1]):
        i = len(current) - i - 1
        if ch in G.VN:
            return next[i: i + len(next) - len(current) + 1]


def get_right_vertex(tree):
    if not tree.children and tree.data in G.VN:
        return tree
    for vert in tree.children[::-1]:
        v = get_right_vertex(vert)
        if v:
            return v


tree = Vertex(choised_ans[0])
for curr, next in zip(choised_ans, choised_ans[1:]):
    changes = get_changes(curr, next)
    v = get_right_vertex(tree)
    v.children = list(map(Vertex, changes))

gt = GraphicalTree(tree)
gt.start()