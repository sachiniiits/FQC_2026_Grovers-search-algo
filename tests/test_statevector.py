"""Our NumPy simulator agrees with qiskit.quantum_info.Statevector."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from grover.circuits import grover_circuit, optimal_iterations, theory_success
from grover.oracles import DEMO_FORMULA, marked_oracle, multi_marked_oracle, sat_oracle
from grover.quantum import runnable
from grover.statevector import fidelity, grover_vector, grover_vector_success, simulate


def qiskit_state_course_order(qc: QuantumCircuit) -> np.ndarray:
    """Qiskit statevector re-indexed so qubit 0 is the most significant bit."""
    return Statevector(qc).reverse_qargs().data


def test_course_order_indexing() -> None:
    qc = QuantumCircuit(3)
    qc.x(0)
    state = simulate(qc)
    assert state[int("100", 2)] == pytest.approx(1)


@pytest.mark.parametrize(
    "n,w,k",
    [(2, "10", 1), (3, "011", 1), (3, "110", 2), (4, "1001", 3), (5, "00111", 2), (6, "101010", 4),
     (7, "1110001", 3), (8, "01100101", 2), (8, "11111111", 5)],
)
def test_gate_level_matches_qiskit(n: int, w: str, k: int) -> None:
    qc = runnable(grover_circuit(marked_oracle(n, w), k, measure=False))
    assert fidelity(simulate(qc), qiskit_state_course_order(qc)) >= 1 - 1e-9


def test_general_unitary_and_cx() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qc.ry(0.3, 1)
    qc.swap(1, 2)
    qc.rzz(0.7, 0, 1)
    assert fidelity(simulate(qc), qiskit_state_course_order(qc)) >= 1 - 1e-9


@pytest.mark.parametrize("n,w,k", [(3, "110", 2), (5, "00111", 4), (8, "01100101", 12)])
def test_vector_form_matches_gate_level(n: int, w: str, k: int) -> None:
    qc = runnable(grover_circuit(marked_oracle(n, w), k, measure=False))
    vec = grover_vector(n, [int(w, 2)], k)
    assert fidelity(simulate(qc), vec.astype(complex)) >= 1 - 1e-9


@pytest.mark.parametrize("n,marked", [(4, [5]), (6, [0, 9, 40]), (10, [1000])])
def test_vector_form_success_matches_theory(n: int, marked: list[int]) -> None:
    N = 2**n
    for k in range(3 * optimal_iterations(N, len(marked)) + 1):
        assert grover_vector_success(n, marked, k) == pytest.approx(theory_success(N, len(marked), k), abs=1e-12)


def test_vector_form_matches_course_notes_n4() -> None:
    notes = [0.062, 0.473, 0.908, 0.961, 0.582, 0.125, 0.020, 0.365, 0.836]
    got = [grover_vector_success(4, [5], k) for k in range(9)]
    assert got == pytest.approx(notes, abs=1e-3)


def test_gate_level_matches_qiskit_with_ancillas() -> None:
    qc = runnable(grover_circuit(sat_oracle(DEMO_FORMULA), 2, measure=False))
    assert fidelity(simulate(qc), qiskit_state_course_order(qc)) >= 1 - 1e-9


def test_gate_level_matches_qiskit_multi_marked() -> None:
    qc = runnable(grover_circuit(multi_marked_oracle(5, ["00001", "11000", "10101"]), 2, measure=False))
    assert fidelity(simulate(qc), qiskit_state_course_order(qc)) >= 1 - 1e-9
