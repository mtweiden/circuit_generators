from qiskit import QuantumCircuit
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL
import numpy as np
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle
import threading

class myThread(threading.Thread):
    def __init__(self, power) -> None:
        super().__init__()
        self.num_q = 2 ** power

    def run(self) -> QuantumCircuit:
        print(self.num_q)
        matrix = np.random.rand(self.num_q, self.num_q)
        matrix = matrix + matrix.conj().T # Make matrix hermitian
        vector = np.random.rand(self.num_q, 1)
        naive_sln = HHL().solve(matrix, vector)
        qiskit_circ = naive_sln.state
        if (type(qiskit_circ) is QuantumCircuit):
            qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])
            num_q = qiskit_circ.num_qubits
            with open(f'temp_qasm_hhl/{num_q}.qasm', 'w') as f:
                f.write(qiskit_circ.qasm())
            # circuit : Circuit = Circuit.from_file(f'temp_qasm/{num_q * 2}.qasm')
            return qiskit_circ
        else:
            with open(f'temp_qasm_hhl/{num_q}.unitary', 'w') as f:
                pickle.dump(qiskit_circ, f)


if __name__ == '__main__':
    '''
    Based on paper to estimate quantum risk analysis. Essentially uses QAE, we can model with
    simple Bernoulli operator with Ry(p).
    Paper is here: 
    https://arxiv.org/pdf/1806.06893.pdf
    '''

    range_low = 1
    range_high = 8
    step = 1
    num_qubits = [x for x in range(range_low, range_high+1, step)]

    threads = [myThread(i) for i in num_qubits]

    qasm_list = [None] * len(num_qubits)

    for i, num_q in enumerate(num_qubits):
        qc = threads[i].start()
        qasm_list[i] = qc

    for t in threads:
        t.join()

    with open(f'finance_hhl_list_{range_low}_{range_high}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)