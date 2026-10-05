"""A small NumPy statevector simulator in course bit order, used to cross-check Qiskit.

The state of n qubits is stored as an array of shape (2,) * n where axis i is qubit i.
Flattening it gives amplitudes indexed by the course bitstring read as a binary number,
qubit 0 being the most significant bit (the leftmost bit of the ket).

Two levels are provided:
- gate level: H, X, Z, multi-controlled gates and general unitaries applied with ``tensordot``,
  plus ``simulate`` which runs a Qiskit circuit gate by gate;
- vector form of Grover: oracle = sign flip on marked indices, diffuser = 2 * mean - amplitude.
  This is the same algorithm without any gates, fast enough for large n.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def zero_state(n: int) -> np.ndarray:
    """|0...0> on n qubits, shape (2,) * n."""
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    return state


def apply_gate(state: np.ndarray, U: np.ndarray, qubits: list[int]) -> np.ndarray:
    """Apply a k-qubit unitary U to the listed qubits.

    U is written in course order: the first listed qubit is the most significant bit of U's index.
    """
    k = len(qubits)
    U = U.reshape((2,) * (2 * k))
    moved = np.tensordot(U, state, axes=(list(range(k, 2 * k)), qubits))
    return np.moveaxis(moved, list(range(k)), qubits)


def apply_controlled(state: np.ndarray, U: np.ndarray, controls: list[int], target: int) -> np.ndarray:
    """Apply a single-qubit U to ``target`` on the part of the state where all controls are 1."""
    if not controls:
        return apply_gate(state, U, [target])
    index: list = [slice(None)] * state.ndim
    for c in controls:
        index[c] = 1
    index = tuple(index)
    out = state.copy()
    sub = state[index]
    sub_target = target - sum(1 for c in controls if c < target)
    out[index] = apply_gate(sub, U, [sub_target])
    return out


def simulate(qc: QuantumCircuit) -> np.ndarray:
    """Run a Qiskit circuit (without classical control) from |0...0> and return the flat state
    in course order. Measurements and barriers are skipped; multi-controlled gates with all-ones
    control state use ``apply_controlled``; any other gate uses its unitary matrix."""
    state = zero_state(qc.num_qubits)
    for inst in qc.data:
        op = inst.operation
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if op.name in ("barrier", "measure"):
            continue
        if op.name == "h":
            state = apply_gate(state, H, qubits)
        elif op.name == "x":
            state = apply_gate(state, X, qubits)
        elif op.name == "z":
            state = apply_gate(state, Z, qubits)
        elif _is_all_ones_controlled(op):
            base = Operator(op.base_gate).data
            state = apply_controlled(state, base, qubits[:-1], qubits[-1])
        else:
            # Qiskit matrices put the first qarg on the least significant bit, so reverse.
            state = apply_gate(state, Operator(op).data, qubits[::-1])
    return state.reshape(-1)


def _is_all_ones_controlled(op) -> bool:
    return (
        isinstance(op, ControlledGate)
        and op.num_qubits - op.num_ctrl_qubits == 1
        and op.ctrl_state == 2**op.num_ctrl_qubits - 1
    )


def grover_vector(n: int, marked: list[int], k: int) -> np.ndarray:
    """Real amplitudes after k Grover iterations, in vector form.

    ``marked`` are course-order indices. The oracle negates marked amplitudes and the diffuser
    2|s><s| - I maps each amplitude a to 2 * mean(a) - a.
    """
    N = 2**n
    a = np.full(N, 1 / math.sqrt(N))
    for _ in range(k):
        a[marked] *= -1
        a = 2 * a.mean() - a
    return a


def grover_vector_success(n: int, marked: list[int], k: int) -> float:
    """Probability of measuring a marked item after k iterations, from the vector form."""
    a = grover_vector(n, marked, k)
    return float(np.sum(a[marked] ** 2))


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2 for normalised state vectors."""
    return float(abs(np.vdot(a, b)) ** 2)
