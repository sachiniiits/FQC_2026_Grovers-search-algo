"""Classical random-order search: honest query counts."""

import numpy as np
import pytest

from grover.classical import (
    QueryCounter,
    classical_expected_queries,
    classical_query_std,
    grover_expected_queries,
    measure_classical,
    random_order_search,
)


def test_counter_counts_calls() -> None:
    counter = QueryCounter(lambda x: int(x == "11"))
    assert [counter(x) for x in ["00", "11", "01"]] == [0, 1, 0]
    assert counter.calls == 3


def test_search_finds_item() -> None:
    rng = np.random.default_rng(0)
    found, calls = random_order_search(lambda x: int(x == "1010"), 4, rng)
    assert found == "1010" and 1 <= calls <= 16


@pytest.mark.parametrize("n", [3, 5, 8])
def test_average_queries_near_n_plus_one_over_two(n: int) -> None:
    N = 2**n
    trials = 2000
    stats = measure_classical(n, trials, seed=n)
    sigma_of_mean = classical_query_std(N) / np.sqrt(trials)
    assert abs(stats.mean - classical_expected_queries(N)) <= 3 * sigma_of_mean
    assert stats.worst <= N


def test_expected_cost_helpers() -> None:
    assert classical_expected_queries(16) == 8.5
    assert classical_expected_queries(15, 4) == 3.2
    assert grover_expected_queries(3, 0.5) == 8.0
