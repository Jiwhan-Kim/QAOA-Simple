import argparse

from qpu import qpu
from simulator import simulator

from circuits import build_random_graph_circuit, expectation
from circuits.build_random_graph_circuit import create_graph

import numpy as np
import rustworkx as rx

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

    gammas = [0.0 for _ in range(n_layers)]
    betas = [0.0 for _ in range(n_layers)]

    for iter in range(n_iterations):
        gammas, betas = iteration_qpu(
            edge_list, gammas, betas, iteration=iter)

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

    if args.env == 'simulator':
        run_on_simulator()

    elif args.env == 'qpu':
        run_on_qpu()

    else:
        print("Wrong Env")
