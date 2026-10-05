"""Every oracle flips the phase of exactly its marked items."""

import pytest

from grover.oracles import (
    DEMO_FORMULA,
    all_strings,
    brute_force_solutions,
    evaluate_formula,
    marked_oracle,
    multi_marked_oracle,
    sat_oracle,
)
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


def test_multi_marked_oracle() -> None:
    ws = ["000", "011", "110"]
    oracle = multi_marked_oracle(3, ws)
    assert [x for x in all_strings(3) if oracle.f(x)] == ws
    assert_matches_truth_table(oracle)
    assert_matches_truth_table(multi_marked_oracle(6, ["000000", "101010", "111111", "010011"]))


def test_multi_marked_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        multi_marked_oracle(3, ["101", "101"])


def test_demo_formula_solutions_by_brute_force() -> None:
    expected = [x for x in all_strings(4) if evaluate_formula(DEMO_FORMULA, x)]
    assert brute_force_solutions(DEMO_FORMULA) == expected == ["0010", "1101"]


def test_sat_oracle_truth_table_and_clean_ancillas() -> None:
    oracle = sat_oracle(DEMO_FORMULA)
    assert oracle.n == 4 and oracle.num_ancillas == len(DEMO_FORMULA) + 1
    assert_matches_truth_table(oracle)


def test_sat_oracle_other_formula() -> None:
    formula = [(1, -2), (2, 3), (-1, -3)]
    oracle = sat_oracle(formula)
    assert list(oracle.marked) == [x for x in all_strings(3) if evaluate_formula(formula, x)]
    assert_matches_truth_table(oracle)
