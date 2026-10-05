"""Run Grover circuits on Qiskit Aer and read the results in course bit order."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from grover.circuits import grover_circuit
from grover.oracles import Oracle


def to_course_order(bitstring: str) -> str:
    """Convert a Qiskit count string (qubit 0 on the right) to course order (qubit 0 on the left)."""
    return bitstring.replace(" ", "")[::-1]


def count_oracle_calls(qc: QuantumCircuit) -> int:
    """Number of oracle blocks in a logical Grover circuit."""
    return qc.count_ops().get("oracle", 0)


def runnable(qc: QuantumCircuit) -> QuantumCircuit:
    """Expand the named oracle and diffuser gates, which Aer cannot run directly."""
    return qc.decompose(gates_to_decompose=["oracle", "diffuser"])


@dataclass(frozen=True)
class GroverResult:
    """Outcome of one Grover run.

    Attributes:
        counts: measurement counts keyed by course-order bitstrings.
        shots: number of shots.
        success_probability: fraction of shots that gave a marked item.
        most_likely: the most frequent bitstring.
        oracle_calls: oracle blocks counted from the circuit.
        found: whether the most likely bitstring is a marked item.
    """

    counts: dict[str, int]
    shots: int
    success_probability: float
    most_likely: str
    oracle_calls: int
    found: bool


def run_grover(oracle: Oracle, k: int, shots: int = 4000, seed: int | None = None) -> GroverResult:
    """Run Grover with k iterations on the ideal Aer simulator."""
    qc = grover_circuit(oracle, k)
    calls = count_oracle_calls(qc)
    raw = AerSimulator().run(runnable(qc), shots=shots, seed_simulator=seed).result().get_counts()
    return _interpret(raw, oracle, shots, calls)


def _interpret(raw: dict[str, int], oracle: Oracle, shots: int, calls: int) -> GroverResult:
    counts = {to_course_order(key): value for key, value in raw.items()}
    hits = sum(counts.get(w, 0) for w in oracle.marked)
    most_likely = max(sorted(counts), key=counts.__getitem__)
    return GroverResult(
        counts=counts,
        shots=shots,
        success_probability=hits / shots,
        most_likely=most_likely,
        oracle_calls=calls,
        found=most_likely in oracle.marked,
    )
