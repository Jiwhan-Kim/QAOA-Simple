import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector


def draw_bloch_sphere():
    qregs = QuantumRegister(2)
    qc = QuantumCircuit(qregs)
    qc.h([0, 1])
    qc.rz(0.5 * np.pi, 0)
    qc.rz(np.pi, 1)

    state = Statevector(qc)
    plot_bloch_multivector(state)
    plt.show()


if __name__ == "__main__":
    draw_bloch_sphere()
