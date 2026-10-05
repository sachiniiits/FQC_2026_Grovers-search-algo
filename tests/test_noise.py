"""No gate may run noiseless by accident, and scale 0 must give ideal results."""

import math

import pytest
from qiskit_aer.noise import NoiseModel, depolarizing_error

from grover.circuits import grover_circuit, optimal_iterations, theory_success
from grover.noise import (
    BASIS_GATES,
    make_noise_model,
    transpile_for_noise,
    uncovered_gates,
)
from grover.oracles import DEMO_FORMULA, marked_oracle, multi_marked_oracle, sat_oracle
from grover.quantum import run_grover

ORACLES = [
    marked_oracle(2, "10"),
    marked_oracle(4, "0110"),
    marked_oracle(6, "110001"),
    multi_marked_oracle(5, ["00000", "10110"]),
    sat_oracle(DEMO_FORMULA),
]


@pytest.mark.parametrize("oracle", ORACLES)
@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_every_transpiled_gate_has_an_error(oracle, scale: float) -> None:
    tqc = transpile_for_noise(grover_circuit(oracle, optimal_iterations(oracle.N, oracle.M)))
    ops = set(tqc.count_ops()) - {"barrier", "measure"}
    assert ops <= set(BASIS_GATES)
    model = make_noise_model(scale)
    assert uncovered_gates(tqc, model) == set()
    assert "measure" in model.noise_instructions


def test_noise_model_rates() -> None:
    probs = {}
    for err in make_noise_model(2.0).to_dict()["errors"]:
        for op in err["operations"]:
            probs[op] = err["probabilities"]
    assert set(probs) == {"rz", "sx", "x", "cx", "measure"}
    assert probs["measure"] == [[0.96, 0.04], [0.08, 0.92]]
    # Depolarizing p leaves the identity with probability 1 - 3p/4 on 1 qubit, 1 - 15p/16 on 2.
    for gate in ["rz", "sx", "x"]:
        assert max(probs[gate]) == pytest.approx(1 - 3 * 0.002 / 4)
    assert max(probs["cx"]) == pytest.approx(1 - 15 * 0.02 / 16)


def test_partial_noise_model_is_refused() -> None:
    partial = NoiseModel(basis_gates=BASIS_GATES)
    partial.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    with pytest.raises(ValueError, match="gates without noise"):
        run_grover(marked_oracle(3, "101"), 2, shots=10, noise=partial, seed=1)


def test_scale_zero_gives_ideal_results() -> None:
    oracle = marked_oracle(4, "1011")
    k = optimal_iterations(16)
    shots = 4000
    p = theory_success(16, 1, k)
    noisy = run_grover(oracle, k, shots=shots, noise=make_noise_model(0.0), seed=9)
    assert abs(noisy.success_probability - p) <= 3 * math.sqrt(p * (1 - p) / shots)
    assert noisy.cnots is not None and noisy.cnots > 0


def test_noise_lowers_success() -> None:
    oracle = marked_oracle(4, "1011")
    k = optimal_iterations(16)
    ideal = run_grover(oracle, k, seed=9).success_probability
    noisy = run_grover(oracle, k, noise=make_noise_model(1.0), seed=9).success_probability
    assert noisy < ideal - 0.2
