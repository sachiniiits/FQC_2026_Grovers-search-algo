"""Diffuser, Grover circuit structure and theory helpers."""

import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from grover.circuits import diffuser, grover_circuit, optimal_iterations, theory_success
from grover.oracles import marked_oracle
from grover.quantum import count_oracle_calls


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_diffuser_is_reflection_about_s(n: int) -> None:
    N = 2**n
    s = np.full(N, 1 / math.sqrt(N))
    target = 2 * np.outer(s, s) - np.eye(N)
    assert Operator(diffuser(n)).equiv(Operator(target))


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_oracle_calls_counted_from_circuit(k: int) -> None:
    qc = grover_circuit(marked_oracle(4, "0110"), k)
    assert count_oracle_calls(qc) == k


def test_barrier_labels() -> None:
    qc = grover_circuit(marked_oracle(3, "111"), 2)
    labels = [inst.operation.label for inst in qc.data if inst.operation.name == "barrier"]
    assert labels == ["psi_0", "psi_1", "psi_2"]


def test_optimal_iterations() -> None:
    assert [optimal_iterations(2**n) for n in range(2, 9)] == [1, 2, 3, 4, 6, 8, 12]
    assert optimal_iterations(64, 4) == 3


def test_theory_success() -> None:
    assert theory_success(4, 1, 1) == pytest.approx(1.0)
    assert theory_success(16, 1, 0) == pytest.approx(1 / 16)
    assert theory_success(16, 1, 3) == pytest.approx(0.9613, abs=1e-4)
