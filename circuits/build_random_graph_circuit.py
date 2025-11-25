import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def create_graph(n_qubits: int):
    graph = rx.PyGraph()

    graph.add_nodes_from(np.arange(0, n_qubits, 1))
    edge_list = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]
    graph.add_edges_from(edge_list)

    # mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
    # plt.show()

    return graph, graph.weighted_edge_list()
    # graph = rx.undirected_gnp_random_graph(n_qubits, 0.3, 1024)
    #
    # for edge_index in graph.edge_indices():
    #     graph.update_edge_by_index(edge_index, 1.0)
    #
    # edge_list = graph.weighted_edge_list()
    #
    # return graph, edge_list


def qaoa_layer(qc, gamma: float, beta: float, n_qubits: int, edge_list: rx.WeightedEdgeList):
    for i, j, _ in edge_list:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)

    for i in range(n_qubits):
        qc.rx(2 * beta, i)


def build_random_graph_circuit(edge_list: rx.WeightedEdgeList, gammas, betas, layers: int, n_qubits: int):
    qregs = QuantumRegister(n_qubits)
    cregs = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qregs, cregs)

    for i in range(n_qubits):
        qc.h(i)

    for layer in range(layers):
        qaoa_layer(qc, gammas[layer], betas[layer], n_qubits, edge_list)

    qc.measure([i for i in range(n_qubits)], [i for i in range(n_qubits)])

    return qc


def maxcut_value(bitstring, edge_list: rx.WeightedEdgeList):
    # bitstring: "010101"
    # bits     : "101010"

    bits = bitstring[::-1]
    value = 0
    for i, j, _ in edge_list:
        if bits[i] != bits[j]:
            value += 1
    return value


def expectation(results, edge_list: rx.WeightedEdgeList):
    list_ret = []

    for result in results:
        total = 0
        shots = 0

        for bitstr, count in result.items():
            total += maxcut_value(bitstr, edge_list) * count
            shots += count
        list_ret.append(total / shots)

    return list_ret


if __name__ == "__main__":
    graph, edge_list = create_graph(10)

    mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
    plt.show()
