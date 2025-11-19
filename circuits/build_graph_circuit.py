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

    return edge_list


def create_random_graph(n_qubits: int):
    pass


def qaoa_layer(qc, gamma: float, beta: float, n_qubits: int, edge_list: list[tuple[int, int, float]]):
    for i, j, weight in edge_list:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)

    for i in range(n_qubits):
        qc.rx(2 * beta, i)


def build_graph_circuit(gammas, betas, layers: int, n_qubits: int):
    edge_list = create_graph(n_qubits)

    qregs = QuantumRegister(n_qubits)
    cregs = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qregs, cregs)

    for i in range(n_qubits):
        qc.h(i)

    for layer in range(layers):
        qaoa_layer(qc, gammas[layer], betas[layer], n_qubits, edge_list)

    qc.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    return edge_list, qc


def maxcut_value(bitstring, edge_list):
    bits = bitstring[::-1]
    value = 0
    for i, j, weight in edge_list:
        if bits[i] != bits[j]:
            value += 1
    return value


def expectation(counts, edge_list):
    total = 0
    shots = 0

    for bitstr, count in counts.items():
        total += maxcut_value(bitstr, edge_list) * count
        shots += count

    return total / shots


if __name__ == "__main__":
    create_graph(5)
