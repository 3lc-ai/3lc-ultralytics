from __future__ import annotations


def flatten(inp: list[tuple[int, int]]) -> list[int]:
    out = []
    for tup in inp:
        out.extend(tup)
    return out


def interleave(a: list[float], b: list[float]) -> list[float]:
    out = []
    for aa, bb in zip(a, b):
        out.append(aa)
        out.append(bb)
    return out
