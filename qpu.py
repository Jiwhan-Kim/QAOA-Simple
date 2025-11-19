from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit import transpile

from circuits import make_bell_circuit


def qpu(quantum_circuit):
    service = QiskitRuntimeService()

    print("Backend:")
    for backend in service.backends():
        print("-", backend.name, "| simulator:",
              backend.configuration().simulator)

    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=5)
    print("Using backend:", backend.name)

    qc = quantum_circuit()
    print(qc)

    tqc = transpile(qc, backend=backend)

    sampler = SamplerV2(mode=backend)

    # When running multiple circuits, just add in the list
    job = sampler.run([tqc], shots=1024)
    job_result = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = job_result[0].data['c'].get_counts()
    print("IBM Backend Result: ", result)

    return result


if __name__ == "__main__":
    qpu(make_bell_circuit)
