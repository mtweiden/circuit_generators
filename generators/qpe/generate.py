from qiskit import QuantumCircuit
from qiskit.circuit.library.phase_estimation import PhaseEstimation
import numpy as np
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle
import threading

def random_circuit(num_q, num_gates) -> QuantumCircuit:
    g = QuantumCircuit(num_q)
    all_qs = np.arange(num_q)
    for _ in range(num_gates):
        i = np.random.randint(2)
        if i == 1:
            q = np.random.choice(all_qs)
            angles = np.random.rand(3) * 2 * np.pi
            g.u(angles[0], angles[1], angles[2], q)
        else:
            qs = np.random.choice(all_qs, size=2, replace=False)
            g.cx(qs[0], qs[1])

    return g

class myThread(threading.Thread):
    def __init__(self, num_q) -> None:
        super().__init__()
        self.num_q = num_q

    def run(self) -> QuantumCircuit:
        num_q = self.num_q
        print(num_q)
        random_unitary = random_circuit(num_q, num_gates=num_q*2)
        qiskit_circ : QuantumCircuit = PhaseEstimation(num_q, random_unitary)
        qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])

        with open(f'temp_qasm/{num_q * 2}.qasm', 'w') as f:
            f.write(qiskit_circ.qasm())
        circuit : Circuit = Circuit.from_file(f'temp_qasm/{num_q * 2}.qasm')
        return circuit


if __name__ == '__main__':
    range_low = 3
    range_high = 9
    step = 5
    num_qubits = [x for x in range(range_low, range_high+1)]

    threads = [myThread(i) for i in num_qubits]

    qasm_list = [None] * len(num_qubits)

    for i, num_q in enumerate(num_qubits):
        qc = threads[i].start()
        qasm_list[i] = qc

    for t in threads:
        t.join()

    with open(f'qpe_list_{range_low}_{range_high}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)