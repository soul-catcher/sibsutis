from __future__ import annotations

import heapq
import itertools
import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Generator, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    'entropy',
    'get_frequencies',
    'gen_subsequences',
    'prepare_text',
    'huffman',
    'encode',
]

H = TypeVar('H', bound=Hashable)
S = TypeVar('S', bound=Sequence)


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(x * math.log2(x) for x in probabilities)


def get_frequencies(collection: Collection[H]) -> dict[H, float]:
    return {k: v / len(collection) for k, v in Counter(collection).items()}


def gen_subsequences(sequence: S, n: int) -> Generator[S]:
    return (sequence[i: i + n] for i in range(len(sequence) - n + 1))


def prepare_text(text: str) -> str:
    return re.sub(r'[^а-я ]', '', text.lower().translate(str.maketrans('ъё\n', 'ье ')))


@dataclass(order=True, frozen=True)
class _Node:
    frequency: float
    elements: tuple[Hashable, ...]

    def __add__(self, other: _Node):
        return _Node(self.frequency + other.frequency, self.elements + other.elements)


def huffman(frequencies: Mapping[H, float], n: int = 2) -> dict[H, list[int]]:
    if len(frequencies) == 1:
        return {next(iter(frequencies)): [0]}
    codes = defaultdict(list)
    heap = [_Node(freq, (elem,)) for elem, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        nodes = [heapq.heappop(heap) for _ in range(min(n, len(heap)))]
        for i, node in enumerate(nodes):
            for e in node.elements:
                codes[e].append(i)
        heapq.heappush(heap, sum(nodes, _Node(0, ())))
    for code in codes.values():
        code.reverse()
    return dict(codes)


def encode(elements: Iterable[H], encoding_table: Mapping[H, Sequence[int]]) -> Iterator[int]:
    return itertools.chain.from_iterable(encoding_table[elem] for elem in elements)
