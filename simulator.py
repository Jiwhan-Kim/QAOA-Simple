from qiskit_aer import AerSimulator
from qiskit import transpile

from circuits import make_bell_circuit


def simulator(qc):
    sim = AerSimulator()

    if __name__ == "__main__":
        print(qc)

    tqc = transpile(qc, sim)

    job = sim.run(tqc, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("Simulator Count: ", counts)

    return counts


if __name__ == "__main__":
    simulator(make_bell_circuit())
