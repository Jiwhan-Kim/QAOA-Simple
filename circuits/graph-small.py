from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def make_small_graph():
    # Creates a small graph

    qregs = QuantumRegister(2)
    cregs = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qregs, cregs)

    return qc
