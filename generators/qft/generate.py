from qiskit.circuit.library.basis_change.qft import QFT
from qiskit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle

if __name__ == '__main__':
    range_low = 4
    range_high = 16
    num_qubits = [x for x in range(range_low, range_high+1, 4)]

    qasm_list = []

    for i, num_q in enumerate(num_qubits):
        qiskit_circ : QuantumCircuit = QFT(num_q)
        qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])

        with open(f'qasm/qft_{num_q}.qasm', 'w') as f:
            f.write(qiskit_circ.qasm())
        circuit : Circuit = Circuit.from_file(f'qasm/qft_{num_q}.qasm')
        qasm_list.append(circuit)

    with open(f'qft_list_{range_low}_{range_high}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)