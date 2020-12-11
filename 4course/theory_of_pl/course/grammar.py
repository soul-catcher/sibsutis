from dataclasses import dataclass


@dataclass(frozen=True)
class Grammar:
    VT: list[str]
    VN: list[str]
    P: dict[str, list[str]]
    S: str
