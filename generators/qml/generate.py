from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
import numpy as np
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle
import threading
from qiskit.transpiler.passes import RemoveBarriers

# def random_two_local(num_qubits, reps = 1):
#     qc = QuantumCircuit(num_qubits)
#     for _ in range(reps):
#         # Create angles for "parametrized gates"
#         angles = np.random.rand(num_qubits, 3) * 2 * np.pi

#         for q in range(num_qubits):
#             qc.u(angles[q][0], angles[q][1], angles[q][2], q)

#         # Now apply CNOTs, linearly

#         for q in range(num_qubits - 1):
#             qc.cx(q, q+1)

#     return qc


def create_ml_circuit(num_qubits, depth):
    '''
    Based on example architecture from Cerezo et. al.
    First encode num_qubits. Then apply parametrized unitary to 2*n gates.
    Connect the bottom n gates to a unitary with 2*n + 1 gates, connect the bottom 
    n + 1 gates to a unitary with 2*n + 3 gates and so on until you hit depth
    '''
    qubit_chunks = [num_qubits] + [(num_qubits + i) for i in range(depth)]
    gate_qubits = []
    last_qubit_size = 0
    last_qubit = 0
    for i,qubit_chunk in enumerate(qubit_chunks):
        if (i == 0): 
            last_qubit_size = qubit_chunk
            continue
        # Add size of gate and qubit range
        gate_qubits.append((qubit_chunk + last_qubit_size, last_qubit))
        last_qubit_size = qubit_chunk
        last_qubit += qubit_chunk


    total_qubits = gate_qubits[-1][0] + gate_qubits[-1][1] 

    circ = QuantumCircuit(total_qubits)

    encoder = ZZFeatureMap(num_qubits, reps=2, insert_barriers=True)
    rand_vals = np.random.rand(encoder.num_parameters) * np.pi
    param_dic = dict(zip(encoder.parameters, rand_vals))
    encoder.assign_parameters(param_dic, inplace=True)

    circ.compose(encoder.decompose(), list(range(num_qubits)), wrap=True,  inplace=True) # Apply encoder on first n qubits

    for gate_size, gate_start_q in gate_qubits:
        twolocal = TwoLocal(gate_size, ['rx', 'rz'], 'cx', reps=2, parameter_prefix="TwoLocal{}".format(gate_size))
        rand_vals = np.random.rand(twolocal.num_parameters) * np.pi
        param_dic = dict(zip(twolocal.parameters, rand_vals))
        # print(twolocal)
        twolocal.assign_parameters(param_dic, inplace=True)
        # twolocal = random_two_local(gate_size, reps=2)
        circ.compose(twolocal.decompose(), list(range(gate_start_q, gate_start_q + gate_size)), wrap=True, inplace=True)

    return circ.decompose()


class myThread(threading.Thread):
    def __init__(self, num_q) -> None:
        super().__init__()
        self.num_q = num_q

    def run(self) -> QuantumCircuit:
        num_q = self.num_q
        print(num_q)
        quantum_circuit = create_ml_circuit(self.num_q, int(np.sqrt(self.num_q)))
        quantum_circuit = RemoveBarriers()(quantum_circuit)
        self.i = quantum_circuit.num_qubits

        with open(f'qasm/qml_{self.i}.qasm', 'w') as f:
            f.write(quantum_circuit.qasm())
        # circuit : Circuit = Circuit.from_file(f'temp_qasm/{i}.qasm')
        print(quantum_circuit.depth())
        print(quantum_circuit.count_ops())

        return quantum_circuit


if __name__ == '__main__':
    range_low = 2
    range_high = 10
    step_size = 1
    num_qubits = [x for x in range(range_low, range_high+1, step_size)]

    threads = [myThread(i) for i in num_qubits]

    qasm_list = [None] * len(num_qubits)

    for i, num_q in enumerate(num_qubits):
        qc = threads[i].start()
        qasm_list[i] = qc

    actual_qubits = []
    for t in threads:
        t.join()
        actual_qubits.append(t.i)

    min_qubit = min(actual_qubits)
    max_qubit = max(actual_qubits)
    with open(f'qml_list_{min_qubit}_{max_qubit}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)
    