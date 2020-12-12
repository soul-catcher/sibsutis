from dataclasses import dataclass


@dataclass(frozen=True)
class Grammar:
    VT: list[str]
    VN: list[str]
    P: dict[str, list[str]]
    S: str

    def make_chains(self, min_length: int, max_length: int) -> list[str]:
        stack = [([], self.S)]
        was_in_stack = set()
        counter = 1
        sequences = []
        while stack:
            prev, sequence = stack.pop()
            prev = prev.copy()
            prev.append(sequence)
            if sequence in was_in_stack:
                continue
            was_in_stack.add(sequence)
            only_term = True
            for i, symbol in enumerate(sequence):
                if symbol in self.VN:
                    only_term = False
                    for elem in self.P[symbol]:
                        scopy = sequence[:i] + elem + sequence[i + 1:]
                        if len(scopy) <= max_length + 3:
                            stack.append((prev, scopy))
            if only_term and min_length <= len(sequence) <= max_length:
                sequences.append(sequence)
                counter += 1
        return sequences
