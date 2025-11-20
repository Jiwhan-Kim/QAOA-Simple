from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit import transpile

from circuits import make_bell_circuit


def qpu(qcs):
    service = QiskitRuntimeService()

    print("Backend:")
    for backend in service.backends():
        print("-", backend.name, "| simulator:",
              backend.configuration().simulator)

    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=5)
    print("Using backend:", backend.name)

    if __name__ == "__main__":
        for qc in qcs:
            print(qc[0])

    tqcs = [transpile(qc, backend=backend) for qc in qcs]

    sampler = SamplerV2(mode=backend)

    # When running multiple circuits, just add in the list
    job = sampler.run(tqcs, shots=1024)
    job_results = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = [job_result.data['c'].get_counts() for job_result in job_results]
    print("IBM Backend Result: ", result)

    return result


if __name__ == "__main__":
    qpu([make_bell_circuit(), make_bell_circuit()])
