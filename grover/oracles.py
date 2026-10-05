"""Phase oracles for Grover's search.

Bit order follows the course: qubit 0 is the top wire and the leftmost bit of every bitstring,
so the string "w0 w1 ... w(n-1)" means qubit i holds w[i].

Every oracle flips the phase of the marked strings, |x> -> (-1)^f(x) |x>, and also exposes its
truth table f(x) so the classical search queries exactly the same function.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class Oracle:
    """A phase oracle on n search qubits, optionally with ancilla qubits.

    Attributes:
        circuit: the oracle circuit. Qubits 0..n-1 are the search register, any further qubits
            are ancillas. Its name is "oracle", so each copy appended to a Grover circuit is one
            countable oracle call.
        n: number of search qubits.
        marked: the marked strings in course bit order.
        kickback_qubit: index (inside ``circuit``) of an ancilla that must be prepared in |->
            before the first call, or None if the oracle needs no such qubit.
    """

    circuit: QuantumCircuit
    n: int
    marked: tuple[str, ...]
    kickback_qubit: int | None = None

    @property
    def N(self) -> int:
        """Number of items, 2^n."""
        return 2**self.n

    @property
    def M(self) -> int:
        """Number of marked items."""
        return len(self.marked)

    @property
    def num_ancillas(self) -> int:
        """Number of qubits beyond the search register."""
        return self.circuit.num_qubits - self.n

    def f(self, x: str) -> int:
        """Truth table: 1 if the n-bit string x (course order) is marked, else 0."""
        return int(x in self.marked)


def all_strings(n: int) -> list[str]:
    """All n-bit strings in increasing order, "00..0" to "11..1"."""
    return ["".join(bits) for bits in product("01", repeat=n)]


def _flip_phase_of(qc: QuantumCircuit, n: int, w: str) -> None:
    """Append gates that flip the phase of |w> only: X where w has 0, multi-controlled Z, X again."""
    zeros = [i for i, bit in enumerate(w) if bit == "0"]
    if zeros:
        qc.x(zeros)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    if zeros:
        qc.x(zeros)


def _check_string(n: int, w: str) -> None:
    if len(w) != n or set(w) - {"0", "1"}:
        raise ValueError(f"marked item {w!r} is not an {n}-bit string")


def marked_oracle(n: int, w: str) -> Oracle:
    """Oracle with exactly one marked string w (course bit order), for n >= 2 qubits."""
    return multi_marked_oracle(n, [w])


def multi_marked_oracle(n: int, ws: list[str]) -> Oracle:
    """Oracle with several marked strings: one phase flip block per marked string."""
    if n < 2:
        raise ValueError("Grover needs at least 2 qubits")
    if len(set(ws)) != len(ws) or not ws:
        raise ValueError("marked strings must be distinct and non-empty")
    qc = QuantumCircuit(n, name="oracle")
    for w in ws:
        _check_string(n, w)
        _flip_phase_of(qc, n, w)
    return Oracle(circuit=qc, n=n, marked=tuple(sorted(ws)))
