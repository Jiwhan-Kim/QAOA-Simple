import argparse

from qpu import qpu
from simulator import simulator

from circuits import build_fixed_graph_circuit, expectation

import numpy as np


N = 20


def run_on_simulator():
    grid = np.linspace(0, np.pi, N)

    best_gamma, best_beta, best_value = None, None, -1
    edge_lists = []
    qcs = []

    for gamma in grid:
        for beta in grid:
            edge_list, qc = build_fixed_graph_circuit([gamma], [beta], 1, 5)

            edge_lists.append(edge_list)
            qcs.append(qc)

    results = simulator(qcs)

    exp_cuts = expectation(results, edge_lists[0])

    for idx, exp_cut in enumerate(exp_cuts):
        print(f"Result with gamma: {grid[idx // N]}, beta: {grid[idx % N]}")
        if exp_cut > best_value:
            best_value = exp_cut
            best_gamma = grid[idx // N]
            best_beta = grid[idx % N]

        print(f"Result: {exp_cut}")

    print("Best: ")
    print("Exp. Cuts: ", best_value)
    print("Gamma: ", best_gamma)
    print("Beta: ", best_beta)


def run_on_qpu():
    grid = np.linspace(0, np.pi, N)

    best_gamma, best_beta, best_value = None, None, -1
    edge_lists = []
    qcs = []

    for gamma in grid:
        for beta in grid:
            edge_list, qc = build_fixed_graph_circuit([gamma], [beta], 1, 5)

            edge_lists.append(edge_list)
            qcs.append(qc)

    results = qpu(qcs)

    exp_cuts = expectation(results, edge_lists[0])

    for idx, exp_cut in enumerate(exp_cuts):
        print(f"Result with gamma: {grid[idx // N]}, beta: {grid[idx % N]}")
        if exp_cut > best_value:
            best_value = exp_cut
            best_gamma = grid[idx // N]
            best_beta = grid[idx % N]

        print(f"Result: {exp_cut}")

    print("Best: ")
    print("Exp. Cuts: ", best_value)
    print("Gamma: ", best_gamma)
    print("Beta: ", best_beta)


parser = argparse.ArgumentParser()
parser.add_argument('--N', required=False, default=20)
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])

args = parser.parse_args()
if __name__ == "__main__":
    N = int(args.N)

    if args.env == 'simulator':
        run_on_simulator()

    elif args.env == 'qpu':
        run_on_qpu()

    else:
        print("Wrong Env")
