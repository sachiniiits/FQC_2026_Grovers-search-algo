"""Grover circuits and the theory formulas they should match.

With sin(theta) = sqrt(M/N), the state after k Grover iterations is
cos((2k+1) theta)|s'> + sin((2k+1) theta)|w>, so P(success) = sin^2((2k+1) theta).
"""

from __future__ import annotations

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from grover.oracles import Oracle


def diffuser(n: int) -> QuantumCircuit:
    """Inversion about the mean, D = H^n (2|0><0| - I) H^n = 2|s><s| - I up to a global phase.

    Gates: H on all, X on all, multi-controlled Z (H, MCX, H on the last qubit), X on all, H on all.
    The circuit equals -(2|s><s| - I), a harmless global phase.
    """
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(oracle: Oracle, k: int, measure: bool = True) -> QuantumCircuit:
    """Full Grover circuit with k iterations.

    The oracle and the diffuser are appended as named gates ("oracle", "diffuser"), so the number
    of oracle calls can be read off the circuit. Barriers are labelled like the course states:
    psi_0 is the uniform superposition |s>, psi_j is the state after j iterations.
    Only the n search qubits are measured.
    """
    n = oracle.n
    q = QuantumRegister(n, "q")
    regs: list = [q]
    if oracle.num_ancillas:
        regs.append(QuantumRegister(oracle.num_ancillas, "a"))
    if measure:
        regs.append(ClassicalRegister(n, "c"))
    qc = QuantumCircuit(*regs)

    if oracle.kickback_qubit is not None:
        qc.x(oracle.kickback_qubit)
        qc.h(oracle.kickback_qubit)
    qc.h(range(n))
    qc.barrier(label="psi_0")

    oracle_gate = oracle.circuit.to_gate(label="Oracle")
    diffuser_gate = diffuser(n).to_gate(label="Diffuser")
    all_qubits = list(range(qc.num_qubits))
    for j in range(1, k + 1):
        qc.append(oracle_gate, all_qubits[: oracle.circuit.num_qubits])
        qc.append(diffuser_gate, all_qubits[:n])
        qc.barrier(label=f"psi_{j}")

    if measure:
        qc.measure(range(n), range(n))
    return qc


def theta(N: int, M: int = 1) -> float:
    """Rotation half-angle, sin(theta) = sqrt(M/N)."""
    return math.asin(math.sqrt(M / N))


def optimal_iterations(N: int, M: int = 1) -> int:
    """Standard choice k_opt = floor((pi/4) sqrt(N/M))."""
    return math.floor(math.pi / 4 * math.sqrt(N / M))


def theory_success(N: int, M: int, k: int) -> float:
    """Probability of measuring a marked item after k iterations, sin^2((2k+1) theta)."""
    return math.sin((2 * k + 1) * theta(N, M)) ** 2
