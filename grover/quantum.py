"""Run Grover circuits on Qiskit Aer and read the results in course bit order."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from grover.circuits import grover_circuit
from grover.noise import cnot_count, transpile_for_noise, uncovered_gates
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
        cnots: CNOT count of the transpiled circuit for noisy runs, None for ideal runs.
    """

    counts: dict[str, int]
    shots: int
    success_probability: float
    most_likely: str
    oracle_calls: int
    found: bool
    cnots: int | None = None


def run_grover(
    oracle: Oracle,
    k: int,
    shots: int = 4000,
    noise: NoiseModel | None = None,
    seed: int | None = None,
) -> GroverResult:
    """Run Grover with k iterations on Aer, ideal (``noise=None``) or with a noise model.

    Oracle calls are counted on the logical circuit. Ideal runs expand the named gates and run
    directly. Noisy runs transpile to the hardware-like basis first and refuse to run if any gate
    in the transpiled circuit has no error attached, so nothing runs noiseless by accident.
    """
    qc = grover_circuit(oracle, k)
    calls = count_oracle_calls(qc)
    if noise is None:
        raw = AerSimulator().run(runnable(qc), shots=shots, seed_simulator=seed).result().get_counts()
        return _interpret(raw, oracle, shots, calls, None)
    tqc = transpile_for_noise(qc)
    missing = uncovered_gates(tqc, noise)
    # A scale-0 model has no errors at all and is ideal by design; any partial model is rejected.
    if missing and noise.noise_instructions:
        raise ValueError(f"gates without noise: {sorted(missing)}")
    sim = AerSimulator(noise_model=noise)
    raw = sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
    return _interpret(raw, oracle, shots, calls, cnot_count(tqc))


def _interpret(
    raw: dict[str, int], oracle: Oracle, shots: int, calls: int, cnots: int | None
) -> GroverResult:
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
        cnots=cnots,
    )
