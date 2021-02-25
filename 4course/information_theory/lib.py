import math
import re
from collections import Counter
from collections.abc import Collection, Generator, Hashable, Iterable, Sequence
from typing import TypeVar

__all__ = [
    'entropy',
    'get_frequencies',
    'gen_subsequences',
    'prepare_text',
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
