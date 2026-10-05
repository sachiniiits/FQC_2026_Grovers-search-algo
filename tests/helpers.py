"""Shared test helpers."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from grover.oracles import Oracle, all_strings


def assert_matches_truth_table(oracle: Oracle) -> None:
    """On every basis input |x>, with ancillas at |0> (and |-> where needed), the oracle must
    return (-1)^f(x) times the same state, so phases are exact and ancillas come back clean."""
    for x in all_strings(oracle.n):
        prep = QuantumCircuit(oracle.circuit.num_qubits)
        ones = [i for i, bit in enumerate(x) if bit == "1"]
        if ones:
            prep.x(ones)
        if oracle.kickback_qubit is not None:
            prep.x(oracle.kickback_qubit)
            prep.h(oracle.kickback_qubit)
        before = Statevector(prep)
        after = before.evolve(oracle.circuit)
        sign = -1 if oracle.f(x) else 1
        assert np.allclose(after.data, sign * before.data, atol=1e-9), x
