"""Every oracle flips the phase of exactly its marked items."""

import pytest

from grover.oracles import all_strings, marked_oracle
from tests.helpers import assert_matches_truth_table


@pytest.mark.parametrize("n", [2, 3, 4])
def test_marked_oracle_every_item(n: int) -> None:
    for w in all_strings(n):
        oracle = marked_oracle(n, w)
        assert [x for x in all_strings(n) if oracle.f(x)] == [w]
        assert_matches_truth_table(oracle)


def test_marked_oracle_larger_n() -> None:
    assert_matches_truth_table(marked_oracle(6, "100110"))


def test_rejects_bad_string() -> None:
    with pytest.raises(ValueError):
        marked_oracle(3, "10")
