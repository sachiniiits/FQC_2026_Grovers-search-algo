"""Regenerate every figure in results/figures, results/summary.md and the generated README blocks.

Run from the repository root:  python scripts\\run_experiments.py

All randomness is seeded, so everything except the timing section is identical between runs.
"""

from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grover.circuits import grover_circuit, optimal_iterations, theory_success, theta  # noqa: E402
from grover.classical import (  # noqa: E402
    classical_expected_queries,
    grover_expected_queries,
    measure_classical,
)
from grover.noise import BASIS_GATES, OPTIMIZATION_LEVEL, P1, P2, make_noise_model  # noqa: E402
from grover.noise import P_READ_0_GIVEN_1, P_READ_1_GIVEN_0  # noqa: E402
from grover.oracles import (  # noqa: E402
    DEMO_FORMULA,
    all_strings,
    formula_to_text,
    marked_oracle,
    multi_marked_oracle,
    sat_oracle,
)
from grover.quantum import run_grover, runnable  # noqa: E402
from grover.statevector import grover_vector, simulate  # noqa: E402

SEED = 7
SHOTS = 4000
CLASSICAL_TRIALS = 500
FIGURES = ROOT / "results" / "figures"
SUMMARY = ROOT / "results" / "summary.md"
README = ROOT / "README.md"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREEN, VIOLET = "#008300", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
    }
)


def log(message: str) -> None:
    print(message, flush=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, metadata={"Software": None})
    plt.close(fig)


def table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fit_slope(N: list[int], y: list[float]) -> float:
    """Slope of log(y) against log(N), the exponent in y ~ N^slope."""
    return float(np.polyfit(np.log(N), np.log(y), 1)[0])


def random_string(rng: np.random.Generator, n: int) -> str:
    return format(int(rng.integers(2**n)), f"0{n}b")


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------


def circuit_figures() -> None:
    """n = 3, marked 101, one iteration: as Oracle and Diffuser blocks, and gate by gate."""
    qc = grover_circuit(marked_oracle(3, "101"), 1)
    for name, circuit in [("circuit_n3_blocks.png", qc), ("circuit_n3.png", runnable(qc))]:
        fig = circuit.draw("mpl", fold=-1, scale=0.8)
        fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)


def correctness(rng: np.random.Generator) -> dict:
    """n = 2..10, one random marked item each (all items for n <= 4)."""
    rows = []
    for n in range(2, 11):
        N = 2**n
        w = random_string(rng, n)
        k = optimal_iterations(N)
        state = simulate(runnable(grover_circuit(marked_oracle(n, w), k, measure=False)))
        p_numpy = float(abs(state[int(w, 2)]) ** 2)
        aer = run_grover(marked_oracle(n, w), k, shots=SHOTS, seed=SEED + n)
        row = {
            "n": n, "N": N, "w": w, "k": k, "theory": theory_success(N, 1, k),
            "numpy": p_numpy, "aer": aer.success_probability, "found": aer.found,
            "calls": aer.oracle_calls, "all_found": None, "all_min": None,
        }
        if n <= 4:
            runs = [run_grover(marked_oracle(n, x), k, shots=SHOTS, seed=SEED) for x in all_strings(n)]
            row["all_found"] = all(r.found for r in runs)
            row["all_min"] = min(r.success_probability for r in runs)
        rows.append(row)
        log(f"  correctness n={n}: w={w} k={k} P_aer={aer.success_probability:.3f}")
    return {"rows": rows}


def queries(correct: dict) -> dict:
    """Classical (measured with the counter) vs Grover (k_opt and fair expected cost)."""
    classical = [measure_classical(n, CLASSICAL_TRIALS, SEED + n) for n in range(2, 17)]
    for c in classical:
        log(f"  classical n={c.n}: mean={c.mean:.1f} worst={c.worst}")
    aer_p = {r["n"]: r["aer"] for r in correct["rows"]}
    grover = []
    for n in range(2, 21):
        N = 2**n
        k = optimal_iterations(N)
        p = theory_success(N, 1, k)
        simulated = n in aer_p
        p_used = aer_p[n] if simulated else p
        grover.append({"n": n, "N": N, "k": k, "p": p_used, "simulated": simulated,
                       "fair": grover_expected_queries(k, p_used)})
    slopes = {
        "classical_mean": fit_slope([2**c.n for c in classical], [c.mean for c in classical]),
        "classical_worst": fit_slope([2**c.n for c in classical], [c.worst for c in classical]),
        "grover_k": fit_slope([g["N"] for g in grover], [g["k"] for g in grover]),
        "grover_fair": fit_slope([g["N"] for g in grover], [g["fair"] for g in grover]),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    Nc = [2**c.n for c in classical]
    ax.plot(Nc, [(N + 1) / 2 for N in Nc], color=MUTED, lw=1, ls=":", label="classical theory (N+1)/2")
    ax.plot(Nc, [c.mean for c in classical], "o-", color=ORANGE,
            label=f"classical average, measured (slope {slopes['classical_mean']:.2f})")
    ax.plot(Nc, [c.worst for c in classical], "s--", color=MAGENTA, ms=5,
            label=f"classical worst case, measured (slope {slopes['classical_worst']:.2f})")
    for key, color, marker, name in [("k", BLUE, "o", "Grover k_opt"), ("fair", AQUA, "D", "Grover fair expected cost")]:
        slope = slopes["grover_k" if key == "k" else "grover_fair"]
        Ng = [g["N"] for g in grover]
        ax.plot(Ng, [g[key] for g in grover], "-", color=color, label=f"{name} (slope {slope:.2f})")
        sim = [g for g in grover if g["simulated"]]
        theo = [g for g in grover if not g["simulated"]]
        z = 4 if key == "k" else 3
        ax.plot([g["N"] for g in sim], [g[key] for g in sim], marker, color=color, ls="none", zorder=z)
        ax.plot([g["N"] for g in theo], [g[key] for g in theo], marker, color=color, ls="none", mfc="white", zorder=z)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of items N")
    ax.set_ylabel("oracle queries")
    ax.set_title("Queries to find one marked item")
    ax.plot([], [], "o", color=MUTED, label="filled: simulated (Aer), open: theory only")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "queries_vs_N.png")
    return {"classical": classical, "grover": grover, "slopes": slopes}


def iterations() -> dict:
    """Success probability vs number of iterations, theory curve and Aer points."""
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, n, w in zip(axes, [4, 6], ["1001", "011010"]):
        N = 2**n
        k_opt = optimal_iterations(N)
        ks = list(range(0, 3 * k_opt + 1))
        sims = [run_grover(marked_oracle(n, w), k, shots=SHOTS, seed=SEED + k).success_probability for k in ks]
        theo = [theory_success(N, 1, k) for k in ks]
        x = np.linspace(0, ks[-1], 400)
        ax.plot(x, np.sin((2 * x + 1) * theta(N)) ** 2, color=MUTED, lw=1.2, label="theory sin^2((2k+1)theta)")
        ax.plot(ks, sims, "o", color=BLUE, label="simulated (Aer)")
        ax.axvline(k_opt, color=ORANGE, lw=1, ls="--", label=f"k_opt = {k_opt}")
        ax.set_title(f"n = {n}, N = {N}, marked {w}")
        ax.set_xlabel("Grover iterations k")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        out[n] = {"w": w, "k_opt": k_opt, "ks": ks, "sim": sims, "theory": theo}
        log(f"  iterations n={n}: done")
    axes[0].set_ylabel("P(marked item)")
    axes[0].plot([], [], color=ORANGE, lw=1, ls="--", label="k_opt")
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [i for i, label in enumerate(labels) if not label.startswith("k_opt =")]
    fig.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "success_vs_iterations.png", metadata={"Software": None})
    plt.close(fig)
    return out


def multiple_marked(rng: np.random.Generator) -> dict:
    """n = 6 with M = 1, 2, 4, 8, 16 marked items."""
    n, N = 6, 64
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    x = np.linspace(0, 10, 400)
    for M, color in zip([1, 2, 4, 8, 16], colors):
        ws = [format(int(i), "06b") for i in sorted(rng.choice(N, M, replace=False))]
        k = optimal_iterations(N, M)
        aer = run_grover(multi_marked_oracle(n, ws), k, shots=SHOTS, seed=SEED + M)
        rows.append({"M": M, "k": k, "ideal_k": math.pi / 4 * math.sqrt(N / M), "theory": theory_success(N, M, k),
                     "aer": aer.success_probability, "classical": classical_expected_queries(N, M)})
        ax1.plot(x, np.sin((2 * x + 1) * theta(N, M)) ** 2, color=color, lw=1.5, label=f"M = {M}")
        ax1.plot([k], [aer.success_probability], "o", color=color, mec="white", mew=1.5, ms=8)
        log(f"  multiple marked M={M}: k={k} P_aer={aer.success_probability:.3f}")
    ax1.set_xlabel("Grover iterations k")
    ax1.set_ylabel("P(some marked item)")
    ax1.set_title("Theory curves, dots = Aer at k_opt")
    ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    Ms = [r["M"] for r in rows]
    xm = np.linspace(1, 16, 200)
    ax2.plot(xm, math.pi / 4 * np.sqrt(N / xm), color=MUTED, lw=1.2, label="(pi/4) sqrt(N/M)")
    ax2.plot(Ms, [r["k"] for r in rows], "o", color=BLUE, label="k_opt used")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("marked items M")
    ax2.set_ylabel("iterations")
    ax2.set_title("Best k shrinks like sqrt(N/M), N = 64")
    ax2.legend(fontsize=8)
    save(fig, "multiple_marked.png")
    return {"rows": rows}


def sat_demo() -> dict:
    """Grover on a SAT formula whose oracle checks the clauses."""
    oracle = sat_oracle(DEMO_FORMULA)
    k = optimal_iterations(oracle.N, oracle.M)
    result = run_grover(oracle, k, shots=SHOTS, seed=SEED)
    xs = all_strings(oracle.n)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [BLUE if x in oracle.marked else "#c3c2b7" for x in xs]
    ax.bar(xs, [result.counts.get(x, 0) / SHOTS for x in xs], color=colors, width=0.8)
    ax.axhline(1 / oracle.N, color=ORANGE, lw=1, ls="--", label="uniform guess 1/16")
    ax.set_xlabel("assignment x1 x2 x3 x4")
    ax.set_ylabel("measured probability")
    ax.set_title(f"Grover on the SAT formula, k = {k} (blue: satisfying assignments)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    save(fig, "sat_histogram.png")
    log(f"  SAT: solutions={list(oracle.marked)} P={result.success_probability:.3f}")
    return {"oracle": oracle, "k": k, "result": result, "qubits": oracle.circuit.num_qubits}


def noise_vs_n(correct: dict) -> dict:
    """Typical noise (scale 1), n = 2..8, same marked items as the correctness run."""
    model = make_noise_model(1.0)
    rows = []
    for r in correct["rows"]:
        if r["n"] > 8:
            continue
        noisy = run_grover(marked_oracle(r["n"], r["w"]), r["k"], shots=SHOTS, noise=model, seed=SEED)
        rows.append({"n": r["n"], "N": r["N"], "k": r["k"], "theory": r["theory"], "ideal": r["aer"],
                     "noisy": noisy.success_probability, "cnots": noisy.cnots, "guess": 1 / r["N"]})
        log(f"  noise n={r['n']}: P_noisy={noisy.success_probability:.3f} CNOTs={noisy.cnots}")
    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.plot(ns, [r["ideal"] for r in rows], "o-", color=BLUE, label="ideal (Aer)")
    ax1.plot(ns, [r["noisy"] for r in rows], "s-", color=ORANGE, label="noisy (Aer, scale 1)")
    ax1.plot(ns, [r["guess"] for r in rows], ":", color=MUTED, label="random guess 1/N")
    ax1.set_yscale("log")
    ax1.set_xlabel("qubits n")
    ax1.set_ylabel("P(marked item)")
    ax1.set_title("Success at k_opt")
    ax1.legend(fontsize=8)
    ax2.plot(ns, [r["cnots"] for r in rows], "o-", color=VIOLET)
    ax2.set_yscale("log")
    ax2.set_xlabel("qubits n")
    ax2.set_ylabel("CNOTs after transpiling")
    ax2.set_title("Circuit size in the rz, sx, x, cx basis")
    save(fig, "noise_vs_n.png")
    return {"rows": rows}


def noise_vs_level(correct: dict) -> dict:
    """n = 3, noise scale 0 to 4."""
    r = next(r for r in correct["rows"] if r["n"] == 3)
    rows = []
    for scale in [0, 0.25, 0.5, 1, 2, 4]:
        res = run_grover(marked_oracle(3, r["w"]), r["k"], shots=SHOTS, noise=make_noise_model(scale), seed=SEED)
        rows.append({"scale": scale, "p": res.success_probability, "cnots": res.cnots})
        log(f"  noise level {scale}: P={res.success_probability:.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([x["scale"] for x in rows], [x["p"] for x in rows], "o-", color=ORANGE, label="noisy (Aer)")
    ax.axhline(r["theory"], color=BLUE, lw=1, ls="--", label="ideal theory")
    ax.axhline(1 / 8, color=MUTED, lw=1, ls=":", label="random guess 1/8")
    ax.set_xlabel("noise scale (1 = typical)")
    ax.set_ylabel("P(marked item)")
    ax.set_ylim(0, 1)
    ax.set_title(f"n = 3, marked {r['w']}, k = {r['k']}")
    ax.legend(fontsize=8)
    save(fig, "noise_vs_level.png")
    return {"rows": rows, "w": r["w"], "k": r["k"], "theory": r["theory"]}


def best_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def simulation_cost() -> dict:
    """Wall-clock time of one Grover iteration on the full 2^n state."""
    rows = []
    sim = AerSimulator(method="statevector")
    for n in range(4, 25, 2):
        qc = QuantumCircuit(n)
        qc.compose(runnable(grover_circuit(marked_oracle(n, "1" * n), 1, measure=False)), inplace=True)
        qc.save_statevector()
        repeats = 3 if n < 22 else 1
        t_aer = best_time(lambda: sim.run(qc).result(), repeats)
        t_np = best_time(lambda: grover_vector(n, [2**n - 1], 1), repeats)
        rows.append({"n": n, "aer": t_aer, "numpy": t_np, "k_opt": optimal_iterations(2**n)})
        log(f"  timing n={n}: aer={t_aer:.4f}s numpy={t_np:.4f}s")
    big = [r for r in rows if r["n"] >= 16]
    growth = {key: float(2 ** fit_slope([2**r["n"] for r in big], [r[key] for r in big])) for key in ["aer", "numpy"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ns = [r["n"] for r in rows]
    ax.plot(ns, [r["aer"] for r in rows], "o-", color=BLUE, label="Qiskit Aer statevector")
    ax.plot(ns, [r["numpy"] for r in rows], "s-", color=AQUA, label="NumPy vector form")
    guide = [rows[-1]["aer"] * 2.0 ** (n - rows[-1]["n"]) for n in ns]
    ax.plot(ns, guide, color=MUTED, lw=1, ls=":", label="doubling per qubit (2^n)")
    ax.set_xticks(ns)
    ax.set_yscale("log")
    ax.set_xlabel("qubits n")
    ax.set_ylabel("seconds per Grover iteration")
    ax.set_title("Simulating one iteration costs ~2^n (machine dependent)")
    ax.legend(fontsize=8)
    save(fig, "simulation_cost.png")
    return {"rows": rows, "growth": growth}


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def build_blocks(c: dict, q: dict, it: dict, mm: dict, sat: dict, nn: dict, nl: dict) -> dict[str, str]:
    blocks = {}
    rows = []
    for r in c["rows"]:
        all_items = "-" if r["all_found"] is None else f"{yes_no(r['all_found'])} (min P {r['all_min']:.3f})"
        rows.append([r["n"], r["N"], f"`{r['w']}`", r["k"], f"{r['theory']:.4f}", f"{r['numpy']:.4f}",
                     f"{r['aer']:.3f}", yes_no(r["found"]), all_items])
    blocks["correctness"] = table(
        ["n", "N", "marked", "k_opt", "theory P", "NumPy P", "Aer P", "top result correct", "every item found (n <= 4)"],
        rows,
    )

    s = q["slopes"]
    cl, gr = q["classical"], q["grover"]
    blocks["slopes"] = "\n".join([
        f"Fitted log-log slopes (queries ~ N^slope), classical for n = {cl[0].n}..{cl[-1].n} "
        f"with {CLASSICAL_TRIALS} searches per n, Grover for n = {gr[0]['n']}..{gr[-1]['n']}:",
        "",
        table(["quantity", "slope"], [
            ["classical average queries (measured)", f"{s['classical_mean']:.3f}"],
            ["classical worst case (measured)", f"{s['classical_worst']:.3f}"],
            ["Grover k_opt (oracle calls)", f"{s['grover_k']:.3f}"],
            ["Grover fair expected cost", f"{s['grover_fair']:.3f}"],
        ]),
    ])

    by_n = {x.n: x for x in cl}
    rows = []
    for g in gr:
        if g["n"] not in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20):
            continue
        cmeas = by_n.get(g["n"])
        cexp = classical_expected_queries(g["N"])
        rows.append([
            g["n"], g["N"], f"{cexp:.1f}", f"{cmeas.mean:.1f}" if cmeas else "-", str(cmeas.worst) if cmeas else "-",
            g["k"], f"{g['p']:.4f}", f"{g['fair']:.2f}", f"{cexp / g['fair']:.1f}",
            "Aer" if g["simulated"] else "theory",
        ])
    blocks["fair"] = table(
        ["n", "N", "classical expected (N+1)/2", "classical measured avg", "classical measured worst",
         "Grover k_opt", "Grover P", "Grover fair cost (k+1)/P", "speedup", "Grover P from"],
        rows,
    )

    rows = []
    for n, d in it.items():
        k_opt = d["k_opt"]
        k_bad = 2 * k_opt
        rows.append([n, k_opt, f"{d['sim'][k_opt]:.3f}", k_bad, f"{d['sim'][k_bad]:.3f}",
                     d["ks"][-1], f"{d['sim'][-1]:.3f}",
                     f"{max(abs(a - b) for a, b in zip(d['sim'], d['theory'])):.3f}"])
    blocks["iterations"] = table(
        ["n", "k_opt", "Aer P at k_opt", "2 k_opt", "Aer P at 2 k_opt", "3 k_opt", "Aer P at 3 k_opt",
         "largest gap between Aer and theory over all k"],
        rows,
    )

    blocks["multi"] = table(
        ["M", "(pi/4) sqrt(N/M)", "k_opt", "theory P", "Aer P", "classical expected (N+1)/(M+1)"],
        [[r["M"], f"{r['ideal_k']:.2f}", r["k"], f"{r['theory']:.4f}", f"{r['aer']:.3f}", f"{r['classical']:.2f}"]
         for r in mm["rows"]],
    )

    o, res = sat["oracle"], sat["result"]
    top = sorted(res.counts, key=res.counts.__getitem__, reverse=True)[:3]
    blocks["sat"] = "\n".join([
        f"- Formula: `{formula_to_text(DEMO_FORMULA)}`",
        f"- Satisfying assignments by brute force (x1 x2 x3 x4): {', '.join('`' + x + '`' for x in o.marked)}",
        f"- Circuit: {o.n} variable qubits, {len(DEMO_FORMULA)} clause ancillas, 1 output qubit in |->, "
        f"{sat['qubits']} qubits in total",
        f"- Grover with M = {o.M}, N = {o.N}: k_opt = {sat['k']}, theory P = {theory_success(o.N, o.M, sat['k']):.4f}, "
        f"Aer P = {res.success_probability:.3f}",
        f"- Three most frequent results: " + ", ".join(f"`{x}` ({res.counts[x] / res.shots:.3f})" for x in top),
    ])

    blocks["noise_n"] = table(
        ["n", "N", "k_opt", "ideal P (Aer)", "noisy P (Aer)", "random guess 1/N", "CNOTs after transpile"],
        [[r["n"], r["N"], r["k"], f"{r['ideal']:.3f}", f"{r['noisy']:.3f}", f"{r['guess']:.3f}", r["cnots"]]
         for r in nn["rows"]],
    )
    blocks["noise_level"] = "\n".join([
        f"n = 3, marked `{nl['w']}`, k = {nl['k']}, ideal theory P = {nl['theory']:.4f}, "
        f"{nl['rows'][0]['cnots']} CNOTs after transpiling.",
        "",
        table(["noise scale", "P(marked item)"], [[r["scale"], f"{r['p']:.3f}"] for r in nl["rows"]]),
    ])
    return blocks


def write_summary(blocks: dict[str, str], cost: dict) -> None:
    timing = table(
        ["n", "Aer s / iteration", "NumPy s / iteration", "k_opt (iterations a full run needs)"],
        [[r["n"], f"{r['aer']:.5f}", f"{r['numpy']:.5f}", r["k_opt"]] for r in cost["rows"]],
    )
    text = f"""# Results summary

Generated by `scripts/run_experiments.py` (seed {SEED}, {SHOTS} shots per run). Do not edit by hand.

Noise model: transpile to basis {", ".join(BASIS_GATES)} (optimization_level {OPTIMIZATION_LEVEL}, fixed
seed_transpiler); depolarizing p1 = {P1} on rz, sx, x, p2 = {P2} on cx; readout P(1|0) = {P_READ_1_GIVEN_0},
P(0|1) = {P_READ_0_GIVEN_1}; all multiplied by the noise scale.

## 1. Correctness (k_opt = floor((pi/4) sqrt(N)))

{blocks["correctness"]}

## 2. Queries vs N

{blocks["slopes"]}

## 3. Fair comparison

{blocks["fair"]}

## 4. Success vs iterations

{blocks["iterations"]}

## 5. Multiple marked items (n = 6, N = 64)

{blocks["multi"]}

## 6. SAT demo

{blocks["sat"]}

## 7. Noise vs n (scale 1)

{blocks["noise_n"]}

## 8. Noise vs level

{blocks["noise_level"]}

## 9. Simulation cost (machine dependent)

Time for one Grover iteration on the full 2^n state, best of several runs. Fitted growth per added
qubit for n >= 16: Aer x{cost["growth"]["aer"]:.2f}, NumPy x{cost["growth"]["numpy"]:.2f}.

{timing}
"""
    SUMMARY.write_text(text, encoding="utf-8", newline="\n")


def update_readme(blocks: dict[str, str]) -> None:
    if not README.exists():
        log("README.md not found, skipping generated blocks")
        return
    text = README.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.S)
        if not pattern.search(text):
            log(f"README.md has no block '{name}'")
            continue
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    README.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log("circuit figures")
    circuit_figures()
    log("1. correctness")
    c = correctness(rng)
    log("2-3. queries vs N")
    q = queries(c)
    log("4. success vs iterations")
    it = iterations()
    log("5. multiple marked items")
    mm = multiple_marked(rng)
    log("6. SAT demo")
    sat = sat_demo()
    log("7. noise vs n")
    nn = noise_vs_n(c)
    log("8. noise vs level")
    nl = noise_vs_level(c)
    log("9. simulation cost")
    cost = simulation_cost()
    blocks = build_blocks(c, q, it, mm, sat, nn, nl)
    write_summary(blocks, cost)
    update_readme(blocks)
    log(f"wrote {SUMMARY.relative_to(ROOT)} and figures in {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
