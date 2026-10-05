"""Ideal Grover runs on Aer match sin^2((2k+1) theta)."""

import math

import pytest

from grover.circuits import grover_circuit, optimal_iterations, theory_success
from grover.oracles import DEMO_FORMULA, all_strings, marked_oracle, multi_marked_oracle, sat_oracle
from grover.quantum import run_grover

SHOTS = 4000


def within_shot_noise(measured: float, p: float, shots: int = SHOTS) -> bool:
    sigma = math.sqrt(p * (1 - p) / shots)
    return abs(measured - p) <= 3 * sigma + 1e-9


@pytest.mark.parametrize(
    "n,w,k",
    [(2, "11", 1), (3, "101", 2), (4, "0110", 1), (4, "0110", 3), (4, "0110", 6), (5, "10011", 4)],
)
def test_ideal_success_matches_theory(n: int, w: str, k: int) -> None:
    result = run_grover(marked_oracle(n, w), k, shots=SHOTS, seed=11)
    assert within_shot_noise(result.success_probability, theory_success(2**n, 1, k))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_finds_marked_item_at_k_opt(n: int) -> None:
    w = "1" * (n - 1) + "0"
    result = run_grover(marked_oracle(n, w), optimal_iterations(2**n), seed=3)
    assert result.found and result.most_likely == w
    assert result.oracle_calls == optimal_iterations(2**n)


def test_ideal_run_handles_named_oracle_gate() -> None:
    oracle = marked_oracle(3, "011")
    assert "oracle" in grover_circuit(oracle, 2).count_ops()
    result = run_grover(oracle, 2, shots=200, seed=5)
    assert sum(result.counts.values()) == 200
    assert result.oracle_calls == 2


@pytest.mark.parametrize("M", [2, 4, 8])
def test_multiple_marked_success_matches_theory(M: int) -> None:
    n = 6
    ws = all_strings(n)[3 :: 64 // M][:M]
    k = optimal_iterations(2**n, M)
    result = run_grover(multi_marked_oracle(n, ws), k, shots=SHOTS, seed=M)
    assert within_shot_noise(result.success_probability, theory_success(2**n, M, k))


def test_sat_grover_amplifies_solutions() -> None:
    oracle = sat_oracle(DEMO_FORMULA)
    k = optimal_iterations(oracle.N, oracle.M)
    result = run_grover(oracle, k, shots=SHOTS, seed=21)
    assert result.oracle_calls == k
    assert within_shot_noise(result.success_probability, theory_success(oracle.N, oracle.M, k))
    top_two = sorted(result.counts, key=result.counts.__getitem__)[-2:]
    assert sorted(top_two) == list(oracle.marked)
