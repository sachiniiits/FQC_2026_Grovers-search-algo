"""Qiskit puts qubit 0 on the right of count strings; the course puts it on the left."""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from grover.quantum import to_course_order


def test_reverses_string() -> None:
    assert to_course_order("001") == "100"
    assert to_course_order("0110") == "0110"
    assert to_course_order("1101") == "1011"


def test_qubit_zero_is_leftmost_in_course_order() -> None:
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure(range(3), range(3))
    counts = AerSimulator().run(qc, shots=10, seed_simulator=1).result().get_counts()
    assert list(counts) == ["001"]
    assert to_course_order("001") == "100"
