from qiskit_aer import AerSimulator
from qiskit import transpile

from circuits import make_bell_circuit


def simulator(qcs):
    sim = AerSimulator()

    if __name__ == "__main__":
        for qc in qcs:
            print(qc)

    tqcs = [transpile(qc, sim) for qc in qcs]

    job = sim.run(tqcs, shots=1024)
    result = job.result()
    counts = result.get_counts()

    if not isinstance(counts, list):
        counts = [counts]

    if __name__ == "__main__":
        print("Simulator Count: ", counts)

    return counts


if __name__ == "__main__":
    simulator([make_bell_circuit()])
