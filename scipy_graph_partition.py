import argparse
from qiskit_ibm_runtime import Session
from qpu_sampler import qpu_sampler, qpu_get_device
from qpu_estimator import qpu_estimator
from sim_sampler import sim_sampler
from sim_estimator import sim_estimator

from circuits import build_qaoa, build_hamiltonian, maxcut_value, expectation, plot_result, print_result

from graphs.long_graph import create_long_graph

from scipy.optimize import minimize

import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt

opt = 'BFGS'
n_qubits = 20
n_layers = 2
n_iterations = 30


def maxcut_cost(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float]):
    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
    result = run([qc])

    return result[0][0]


def maxcut_grad(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float]):
    params = []
    for idx in range(2 * n_layers):
        plus = thetas.copy()
        minus = thetas.copy()

        plus[idx] += np.pi / 2.0
        minus[idx] -= np.pi / 2.0

        params.append(plus)
        params.append(minus)

    results = run([build_qaoa(
        edge_list, param[:n_layers], param[n_layers:], n_layers, n_qubits)
        for param in params])

    exp_cut = [result[0] for result in results]
    # exp_cut = expectation(result, edge_list)

    plus = exp_cut[0:4*n_qubits:2]
    minus = exp_cut[1:4*n_qubits:2]

    grad = [0.5 * (p - m) for p, m in zip(plus, minus)]
    return grad


def get_cost_only(thetas: list[float],
                  run,
                  edge_list: rx.WeightedEdgeList):
    cost = maxcut_cost(run, edge_list, thetas)
    print(f"Cost: {cost}")

    return cost


def get_cost_grad(thetas: list[float],
                  run,
                  edge_list: rx.WeightedEdgeList):
    cost = maxcut_cost(run, edge_list, thetas)
    grad = [grad for grad in maxcut_grad(run, edge_list, thetas)]

    print(f"Cost: {cost}")
    print(f"Grad: {grad}")

    # return cost
    return cost, grad


def simulator():
    graph, edge_list = create_long_graph(n_qubits)
    if args.draw != 0:
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.show()

    thetas = 2 * np.pi * np.random.rand(2 * n_layers)
    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    # Training
    if opt == 'BFGS':
        minimizer = minimize(get_cost_grad, thetas, args=(
            lambda qcs: sim_estimator(
                qcs, hamiltonian, 4096
            ),
            edge_list),
            method="BFGS", jac=True, options={"maxiter": n_iterations})
    else:
        minimizer = minimize(get_cost_only, thetas, args=(
            lambda qcs: sim_estimator(
                qcs, hamiltonian, 4096
            ),
            edge_list),
            method="COBYLA", options={"maxiter": n_iterations})

    print(minimizer)
    thetas = minimizer.x

    # Inference
    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
    print("Count Gate:")
    print(qc.count_ops())

    results = sim_sampler([qc])

    plot_results = plot_result(results, edge_list)
    print_result(plot_results)


def qpu():
    backend = qpu_get_device()

    graph, edge_list = create_long_graph(n_qubits)
    if args.draw != 0:
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.show()

    thetas = 2 * np.pi * np.random.rand(2 * n_layers)
    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    with Session(backend=backend) as session:
        # Training
        if opt == 'BFGS':
            minimizer = minimize(get_cost_grad, thetas, args=(
                lambda qcs: qpu_estimator(
                    backend, session, qcs, hamiltonian, 4096),
                edge_list),
                method="BFGS", jac=True, options={"maxiter": n_iterations})
        else:
            minimizer = minimize(get_cost_only, thetas, args=(
                lambda qcs: qpu_estimator(
                    backend, session, qcs, hamiltonian, 4096),
                edge_list),
                method="COBYLA", options={"maxiter": n_iterations})

        print(minimizer)
        thetas = minimizer.x

        # Inference
        qc = build_qaoa(
            edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
        print("Count Gate:")
        print(qc.count_ops())

        results = qpu_sampler(backend, session, [qc], 4096)

    plot_results = plot_result(results, edge_list)
    print_result(plot_results)


parser = argparse.ArgumentParser()
parser.add_argument('--draw', required=False, default=0)
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])

parser.add_argument('--qubits', required=False, type=int, default=15)
parser.add_argument('--layers', required=False, type=int, default=2)
parser.add_argument('--iters', required=False, type=int, default=30)
parser.add_argument('--opt', required=False, default='BFGS',
                    choices=['BFGS', 'COBYLA'])
args = parser.parse_args()


def main():
    if args.env == "simulator":
        simulator()
    elif args.env == "qpu":
        qpu()
    else:
        return


if __name__ == "__main__":
    n_qubits = args.qubits
    n_layers = args.layers
    n_iterations = args.iters
    opt = args.opt
    main()
