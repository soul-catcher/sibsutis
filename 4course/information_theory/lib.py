import math
from collections import Counter
from typing import Collection, Iterable, Sequence


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(x * math.log2(x) for x in probabilities)


def get_frequencies(sequence: Collection[str]) -> dict[str, float]:
    return {k: v / len(sequence) for k, v in Counter(sequence).items()}


def gen_subsequences(sequence: Sequence[str], n: int) -> Iterable[str]:
    for i in range(len(sequence) - n + 1):
        yield sequence[i: i + n]
