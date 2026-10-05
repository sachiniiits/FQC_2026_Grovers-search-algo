"""Hardware-like transpilation and the matching Aer noise model.

Grover's multi-controlled gates mean nothing to a noise model until they are broken down into a
hardware basis, so noisy runs always use a circuit transpiled to ``BASIS_GATES``, and the noise
model attaches an error to every one of those gates. ``uncovered_gates`` lists any gate that would
run noiseless, and the tests require it to be empty.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS_GATES = ["rz", "sx", "x", "cx"]
ONE_QUBIT_GATES = ["rz", "sx", "x"]
OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 2024

P1 = 0.001
P2 = 0.01
P_READ_1_GIVEN_0 = 0.02
P_READ_0_GIVEN_1 = 0.04


def transpile_for_noise(qc: QuantumCircuit) -> QuantumCircuit:
    """Transpile to the hardware-like basis with fixed settings, so CNOT counts are reproducible."""
    return transpile(
        qc,
        basis_gates=BASIS_GATES,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )


def make_noise_model(scale: float = 1.0) -> NoiseModel:
    """Depolarizing noise on every basis gate plus readout error, all multiplied by ``scale``.

    1-qubit gates (rz, sx, x): depolarizing p1 = 0.001 x scale.
    CNOT: 2-qubit depolarizing p2 = 0.01 x scale.
    Readout: P(1|0) = 0.02 x scale, P(0|1) = 0.04 x scale.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative")
    model = NoiseModel(basis_gates=BASIS_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(P1 * scale, 1), ONE_QUBIT_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(P2 * scale, 2), ["cx"])
    p10 = P_READ_1_GIVEN_0 * scale
    p01 = P_READ_0_GIVEN_1 * scale
    model.add_all_qubit_readout_error(ReadoutError([[1 - p10, p10], [p01, 1 - p01]]))
    return model


def cnot_count(qc: QuantumCircuit) -> int:
    """Number of CNOTs in a (transpiled) circuit."""
    return qc.count_ops().get("cx", 0)


def uncovered_gates(qc: QuantumCircuit, model: NoiseModel) -> set[str]:
    """Operations in ``qc`` that have no error in ``model`` (barriers excluded).

    ``measure`` counts as covered only if the model has a readout error.
    """
    used = set(qc.count_ops()) - {"barrier"}
    return used - set(model.noise_instructions)
