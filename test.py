from circuits import build_qaoa, build_hamiltonian
import rustworkx as rx
import numpy as np

from sim_estimator import sim_estimator


def main():
    graph = rx.PyGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edges_from([(0, 1, 1.0)])

    edge_list = graph.weighted_edge_list()
    hamiltonian = build_hamiltonian(2, edge_list)
    qc = build_qaoa(edge_list, [
                    np.pi / 2.0], [np.pi / 8.0], 1, 2)

    result = sim_estimator([qc], hamiltonian, 1024)
    print(result)

    return result[0][0]


if __name__ == "__main__":
    main()
