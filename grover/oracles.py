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

Formula = list[tuple[int, ...]]
"""A CNF formula in DIMACS style: each clause is a tuple of literals, literal +j means x_j and
-j means NOT x_j, with variables numbered 1..n. Variable x_j lives on qubit j - 1."""

DEMO_FORMULA: Formula = [(1, 2, 3), (-1, -3), (1, -2), (3, 4), (2, -4)]
"""The SAT instance used in the experiments and the notebook (4 variables, 5 clauses)."""


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


def num_variables(formula: Formula) -> int:
    """Number of variables, the largest variable index used in the formula."""
    return max(abs(lit) for clause in formula for lit in clause)


def evaluate_formula(formula: Formula, x: str) -> bool:
    """Evaluate a CNF formula on the assignment x, where x[j - 1] is the value of x_j."""
    return all(any((x[abs(lit) - 1] == "1") == (lit > 0) for lit in clause) for clause in formula)


def brute_force_solutions(formula: Formula) -> list[str]:
    """All satisfying assignments, found by trying every assignment."""
    n = num_variables(formula)
    return [x for x in all_strings(n) if evaluate_formula(formula, x)]


def formula_to_text(formula: Formula) -> str:
    """Readable form, e.g. "(x1 OR NOT x2) AND (x2 OR x3)"."""
    def literal(lit: int) -> str:
        return f"x{lit}" if lit > 0 else f"NOT x{-lit}"

    return " AND ".join("(" + " OR ".join(literal(lit) for lit in clause) + ")" for clause in formula)


def sat_oracle(formula: Formula) -> Oracle:
    """Oracle that checks a CNF formula gate by gate, without knowing its solutions.

    Qubits: n variable qubits, then one ancilla per clause, then one output qubit that Grover's
    circuit prepares in |->. Each clause ancilla is set to the OR of its literals using
    De Morgan (OR = NOT AND of negations): X on positive literals, multi-controlled X onto the
    ancilla, X on the ancilla, then undo the literal X gates. A multi-controlled X from all clause
    ancillas onto the |-> qubit kicks back the phase (-1)^f(x), as in lecture 08. The clause
    ancillas are then uncomputed, so they return to |0>.

    The ``marked`` field is filled by brute force only so the classical side and the tests can use
    the truth table; the circuit itself is built from the clauses alone.
    """
    n = num_variables(formula)
    c = len(formula)
    out = n + c
    qc = QuantumCircuit(n + c + 1, name="oracle")

    def compute_clause(j: int) -> None:
        clause = formula[j]
        flips = [lit - 1 for lit in clause if lit > 0]
        controls = [abs(lit) - 1 for lit in clause]
        if flips:
            qc.x(flips)
        qc.mcx(controls, n + j)
        qc.x(n + j)
        if flips:
            qc.x(flips)

    for j in range(c):
        compute_clause(j)
    qc.mcx(list(range(n, n + c)), out)
    for j in reversed(range(c)):
        compute_clause(j)

    return Oracle(circuit=qc, n=n, marked=tuple(brute_force_solutions(formula)), kickback_qubit=out)
