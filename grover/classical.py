"""Classical unstructured search with an honest query counter, and expected-cost formulas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from grover.oracles import all_strings, marked_oracle


class QueryCounter:
    """Wraps f and counts every call. One call is one classical query."""

    def __init__(self, f: Callable[[str], int]) -> None:
        self.f = f
        self.calls = 0

    def __call__(self, x: str) -> int:
        self.calls += 1
        return self.f(x)


def random_order_search(f: Callable[[str], int], n: int, rng: np.random.Generator) -> tuple[str | None, int]:
    """Query the 2^n items in a random order until f(x) = 1.

    The search stops on the first marked item it sees; it does not assume how many items are
    marked, so the item it returns has been checked. Returns (item or None, queries used).
    """
    counter = QueryCounter(f)
    items = all_strings(n)
    for index in rng.permutation(len(items)):
        if counter(items[index]):
            return items[index], counter.calls
    return None, counter.calls


@dataclass(frozen=True)
class ClassicalStats:
    """Measured query counts of the classical search for one n."""

    n: int
    trials: int
    mean: float
    std: float
    worst: int


def measure_classical(n: int, trials: int, seed: int) -> ClassicalStats:
    """Run random-order search ``trials`` times, each with a fresh random marked item and order.

    The function queried is the truth table of the same oracle object used on the quantum side.
    """
    rng = np.random.default_rng(seed)
    items = all_strings(n)
    queries = []
    for _ in range(trials):
        w = items[rng.integers(len(items))]
        found, calls = random_order_search(marked_oracle(n, w).f, n, rng)
        if found != w:
            raise RuntimeError(f"search returned {found!r}, expected {w!r}")
        queries.append(calls)
    q = np.array(queries)
    return ClassicalStats(n=n, trials=trials, mean=float(q.mean()), std=float(q.std(ddof=1)), worst=int(q.max()))


def classical_expected_queries(N: int, M: int = 1) -> float:
    """Expected queries of random-order search until the first of M marked items: (N+1)/(M+1)."""
    return (N + 1) / (M + 1)


def classical_query_std(N: int) -> float:
    """Standard deviation of the query count for one marked item (uniform on 1..N)."""
    return float(np.sqrt((N**2 - 1) / 12))


def classical_worst_case(N: int, M: int = 1) -> int:
    """Worst case of the search above: every unmarked item comes first, N - M + 1 queries."""
    return N - M + 1


def grover_expected_queries(k: int, p: float) -> float:
    """Fair expected cost of Grover: each attempt makes k oracle calls plus one classical query to
    check the candidate, and succeeds with probability p, so on average 1/p attempts."""
    return (k + 1) / p
