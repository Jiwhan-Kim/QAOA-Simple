from qpu import qpu
from simulator import simulator

from circuits import build_graph_circuit, expectation

import numpy as np

if __name__ == "__main__":
    # run = qpu
    run = simulator

    grid = np.linspace(0, np.pi, 20)
    best_gamma, best_beta, best_value = None, None, -1

    for gamma in grid:
        for beta in grid:
            print(f"Running with - gamma: {gamma}, beta: {beta}")
            edge_list, qc = build_graph_circuit([gamma], [beta], 1, 5)
            result = run(qc)

            exp_cut = expectation(result, edge_list)

            if exp_cut > best_value:
                best_value = exp_cut
                best_gamma = gamma
                best_beta = beta

            print(f"Result: {exp_cut}")

    print("Best: ")
    print("Cuts: ", best_value)
    print("Gamma: ", best_gamma)
    print("Beta: ", best_beta)
