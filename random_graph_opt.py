import argparse

from qpu import qpu
from simulator import simulator

from circuits import build_random_graph_circuit, expectation
from circuits.build_random_graph_circuit import create_graph

import numpy as np
import rustworkx as rx

# Optional: use SciPy optimizers when available
try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except Exception:
    exit()

n_qubit = 5
n_layers = 10
n_iterations = 100

# Learning rate for gradient-ascent (maximize expected cut)
LEARNING_RATE = 0.999


def _clip_angles(gammas: list[float], betas: list[float]):
    # Wrap into conventional QAOA domains using periodicity:
    # gamma has period pi, beta has period pi/2 in this parameterization
    g = [float(np.mod(x, np.pi)) for x in gammas]
    b = [float(np.mod(x, np.pi / 2)) for x in betas]
    return g, b


def _pack_params(gammas: list[float], betas: list[float]) -> np.ndarray:
    return np.array(list(gammas) + list(betas), dtype=float)


def _unpack_params(x: np.ndarray) -> tuple[list[float], list[float]]:
    g = list(map(float, x[:n_layers]))
    b = list(map(float, x[n_layers:2 * n_layers]))
    return g, b


def _clip_params_vector(x: np.ndarray) -> np.ndarray:
    gammas, betas = _unpack_params(x)
    gammas, betas = _clip_angles(gammas, betas)
    return _pack_params(gammas, betas)


def _objective_sim(edge_list: rx.WeightedEdgeList, x: np.ndarray) -> float:
    # SciPy performs minimization; we want to maximize expected cut.
    # So return negative expectation value.
    x = _clip_params_vector(x)
    gammas, betas = _unpack_params(x)
    qc = build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit)
    result = simulator([qc])
    f = expectation(result, edge_list)[0]
    return -float(f)


def _gradient_sim(edge_list: rx.WeightedEdgeList, x: np.ndarray) -> np.ndarray:
    # Parameter-shift gradient for both gamma and beta.
    # Returns gradient of negative objective (i.e., -f), shape (2 * n_layers,)
    x = _clip_params_vector(x)
    gammas, betas = _unpack_params(x)

    qcs = []
    # Base circuit (unused for gradient, but keep aligned with pattern)
    qcs.append(build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit))

    # Gammas +/-
    for layer in range(n_layers):
        gammas_plus = [g if i != layer else (
            g + np.pi / 4) for i, g in enumerate(gammas)]
        qcs.append(build_random_graph_circuit(
            edge_list, gammas_plus, betas, n_layers, n_qubit))
        gammas_minus = [g if i != layer else (
            g - np.pi / 4) for i, g in enumerate(gammas)]
        qcs.append(build_random_graph_circuit(
            edge_list, gammas_minus, betas, n_layers, n_qubit))

    # Betas +/-
    for layer in range(n_layers):
        betas_plus = [b if i != layer else (
            b + np.pi / 4) for i, b in enumerate(betas)]
        qcs.append(build_random_graph_circuit(
            edge_list, gammas, betas_plus, n_layers, n_qubit))
        betas_minus = [b if i != layer else (
            b - np.pi / 4) for i, b in enumerate(betas)]
        qcs.append(build_random_graph_circuit(
            edge_list, gammas, betas_minus, n_layers, n_qubit))

    results = simulator(qcs)
    exp_cuts = expectation(results, edge_list)
    # exp_cuts[0] corresponds to the base circuit's value
    exp_cuts = exp_cuts[1:]

    f_gamma_plus = [exp_cut for exp_cut in exp_cuts[0:n_layers * 2:2]]
    f_gamma_minus = [exp_cut for exp_cut in exp_cuts[1:n_layers * 2:2]]
    f_beta_plus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2:n_layers * 4:2]]
    f_beta_minus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2 + 1:n_layers * 4:2]]

    grad_gammas = [(f_gamma_plus[i] - f_gamma_minus[i])
                   for i in range(n_layers)]
    grad_betas = [(f_beta_plus[i] - f_beta_minus[i]) for i in range(n_layers)]

    # We return gradient of -f
    return -_pack_params(grad_gammas, grad_betas)


def iteration_simulation(edge_list: rx.WeightedEdgeList,
                         gammas: list[float],
                         betas: list[float],
                         iteration: int = 0):
    qcs = []

    qc = build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit)
    qcs.append(qc)

    # Gammas (parameter-shift with s = pi/4 for exp(-i gamma ZZ))
    for layer in range(n_layers):
        gammas_plus = [gamma if idx != layer else (
            gamma + np.pi / 4) for idx, gamma in enumerate(gammas)]
        qc = build_random_graph_circuit(
            edge_list, gammas_plus, betas, n_layers, n_qubit)
        qcs.append(qc)

        gammas_minus = [gamma if idx != layer else (
            gamma - np.pi / 4) for idx, gamma in enumerate(gammas)]
        qc = build_random_graph_circuit(
            edge_list, gammas_minus, betas, n_layers, n_qubit)

        qcs.append(qc)

    # Betas (parameter-shift with s = pi/4 for exp(-i beta X))
    for layer in range(n_layers):
        betas_plus = [beta if idx != layer else (
            beta + np.pi / 4) for idx, beta in enumerate(betas)]
        qc = build_random_graph_circuit(
            edge_list, gammas, betas_plus, n_layers, n_qubit)
        qcs.append(qc)

        betas_minus = [beta if idx != layer else (
            beta - np.pi / 4) for idx, beta in enumerate(betas)]
        qc = build_random_graph_circuit(
            edge_list, gammas, betas_minus, n_layers, n_qubit)
        qcs.append(qc)

    results = simulator(qcs)
    exp_cuts = expectation(results, edge_list)

    print(f"Iteration {iteration}'s Exp-cut: {exp_cuts[0]}")
    exp_cuts = exp_cuts[1:]

    f_gamma_plus = [exp_cut for exp_cut in exp_cuts[0:n_layers * 2:2]]
    f_gamma_minus = [exp_cut for exp_cut in exp_cuts[1:n_layers * 2:2]]

    f_beta_plus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2:n_layers * 4:2]]
    f_beta_minus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2 + 1:n_layers * 4:2]]

    # Parameter-shift gradient for our parametrization:
    # U_C = exp(-i gamma ZZ), U_M = exp(-i beta X)
    # => ∂/∂gamma f = f(gamma+π/4) - f(gamma-π/4)
    #    ∂/∂beta  f = f(beta +π/4) - f(beta -π/4)
    grad_gammas = [(f_gamma_plus[i] - f_gamma_minus[i])
                   for i in range(n_layers)]
    grad_betas = [(f_beta_plus[i] - f_beta_minus[i])
                  for i in range(n_layers)]

    print(f"grad_gammas: {grad_gammas}, grad_betas: {grad_betas}")

    # Gradient-ascent with a small constant learning rate
    gammas = [(gamma + (LEARNING_RATE ** iteration) * grad_gammas[i])
              for i, gamma in enumerate(gammas)]
    betas = [(beta + (LEARNING_RATE ** iteration) * grad_betas[i])
             for i, beta in enumerate(betas)]

    # Keep angles in conventional QAOA ranges
    gammas, betas = _clip_angles(gammas, betas)

    print(f"gammas: {gammas}, betas: {betas}")

    return gammas, betas


def run_on_simulator():
    _, edge_list = create_graph(n_qubit)

    # Initial parameters
    x0 = np.zeros(2 * n_layers, dtype=float)

    if args.optimizer != 'manual' and _SCIPY_AVAILABLE:
        method = args.optimizer
        print(f"Using SciPy optimizer: {method}")

        # Bounds to keep angles in conventional domains where supported
        bounds = [(0.0, np.pi)] * n_layers + [(0.0, np.pi / 2.0)] * n_layers

        # Gradient available for gradient-based methods
        uses_jac = method in {"L-BFGS-B", "BFGS", "CG", "SLSQP", "TNC"}
        options = {"maxiter": args.maxiter}
        res = minimize(
            fun=lambda x: _objective_sim(edge_list, x),
            x0=x0,
            jac=(lambda x: _gradient_sim(edge_list, x)) if uses_jac else None,
            method=method,
            bounds=bounds if method in {"L-BFGS-B", "SLSQP", "TNC"} else None,
            options=options,
        )
        x_opt = _clip_params_vector(res.x)
        gammas, betas = _unpack_params(x_opt)
    else:
        if args.optimizer != 'manual' and not _SCIPY_AVAILABLE:
            print("SciPy not available. Falling back to manual gradient ascent.")
        else:
            print("Using manual gradient-ascent optimizer.")

        gammas = [0.0 for _ in range(n_layers)]
        betas = [0.0 for _ in range(n_layers)]
        for iter in range(n_iterations):
            gammas, betas = iteration_simulation(
                edge_list, gammas, betas, iteration=iter)

    qc = build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit)
    result = simulator([qc])
    exp_cut = expectation(result, edge_list)
    print("Final Exp-cut: ", exp_cut)
    print("Gammas: ", gammas)
    print("Betas: ", betas)


def iteration_qpu(edge_list: rx.WeightedEdgeList,
                  gammas: list[float] = None,
                  betas: list[float] = None,
                  iteration: int = 0):
    if gammas is None:
        gammas = [0.0 for _ in range(n_layers)]
    if betas is None:
        betas = [0.0 for _ in range(n_layers)]
    qcs = []

    qc = build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit)
    qcs.append(qc)

    # Gammas (parameter-shift with s = pi/4 for exp(-i gamma ZZ))
    for layer in range(n_layers):
        gammas_plus = [gamma if idx != layer else (
            gamma + np.pi / 4) for idx, gamma in enumerate(gammas)]
        qc = build_random_graph_circuit(
            edge_list, gammas_plus, betas, n_layers, n_qubit)
        qcs.append(qc)

        gammas_minus = [gamma if idx != layer else (
            gamma - np.pi / 4) for idx, gamma in enumerate(gammas)]
        qc = build_random_graph_circuit(
            edge_list, gammas_minus, betas, n_layers, n_qubit)

        qcs.append(qc)

    # Betas (parameter-shift with s = pi/4 for exp(-i beta X))
    for layer in range(n_layers):
        betas_plus = [beta if idx != layer else (
            beta + np.pi / 4) for idx, beta in enumerate(betas)]
        qc = build_random_graph_circuit(
            edge_list, gammas, betas_plus, n_layers, n_qubit)
        qcs.append(qc)

        betas_minus = [beta if idx != layer else (
            beta - np.pi / 4) for idx, beta in enumerate(betas)]
        qc = build_random_graph_circuit(
            edge_list, gammas, betas_minus, n_layers, n_qubit)
        qcs.append(qc)

    results = qpu(qcs)
    exp_cuts = expectation(results, edge_list)

    print(f"Iteration {iteration}'s Exp-cut: {exp_cuts[0]}")
    exp_cuts = exp_cuts[1:]

    f_gamma_plus = [exp_cut for exp_cut in exp_cuts[0:n_layers * 2:2]]
    f_gamma_minus = [exp_cut for exp_cut in exp_cuts[1:n_layers * 2:2]]

    f_beta_plus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2:n_layers * 4:2]]
    f_beta_minus = [
        exp_cut for exp_cut in exp_cuts[n_layers * 2 + 1:n_layers * 4:2]]

    grad_gammas = [(f_gamma_plus[i] - f_gamma_minus[i])
                   for i in range(n_layers)]
    # BUGFIX: index range; both f_beta_* have length n_layers
    grad_betas = [(f_beta_plus[i] - f_beta_minus[i])
                  for i in range(n_layers)]

    # Gradient-ascent update
    gammas = [(gamma + LEARNING_RATE * grad_gammas[i])
              for i, gamma in enumerate(gammas)]
    betas = [(beta + LEARNING_RATE * grad_betas[i])
             for i, beta in enumerate(betas)]

    gammas, betas = _clip_angles(gammas, betas)

    return gammas, betas


def run_on_qpu():
    graph, edge_list = create_graph(n_qubit)

    # For QPU runs, keep the manual parameter-shift ascent to limit shots
    gammas = [0.0 for _ in range(n_layers)]
    betas = [0.0 for _ in range(n_layers)]

    for iter in range(n_iterations):
        gammas, betas = iteration_qpu(edge_list, gammas, betas, iteration=iter)

    qc = build_random_graph_circuit(
        edge_list, gammas, betas, n_layers, n_qubit)
    result = qpu([qc])
    exp_cut = expectation(result, edge_list)
    print("Final Exp-cut: ", exp_cut)
    print("Gammas: ", gammas)
    print("Betas: ", betas)


parser = argparse.ArgumentParser()
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])
parser.add_argument('--qubit', required=False, type=int, default=5)
parser.add_argument('--layers', required=False, type=int, default=10)
parser.add_argument('--iters', required=False, type=int, default=100)
parser.add_argument('--lr', required=False, type=float, default=LEARNING_RATE)
parser.add_argument('--optimizer', required=False, type=str,
                    default='COBYLA',
                    choices=['manual', 'COBYLA', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'BFGS', 'CG', 'Powell', 'TNC'])
parser.add_argument('--maxiter', required=False, type=int, default=200)

args = parser.parse_args()
if __name__ == "__main__":
    n_qubit = args.qubit
    n_layers = args.layers
    n_iterations = args.iters
    LEARNING_RATE = args.lr

    if args.env == 'simulator':
        run_on_simulator()
    elif args.env == 'qpu':
        run_on_qpu()
    else:
        print("Wrong Env")
