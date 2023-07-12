from qiskit import QuantumCircuit
from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import AmplitudeEstimation
from qiskit.primitives import Sampler
import numpy as np
from qiskit.compiler.transpiler import transpile
from bqskit import Circuit
import pickle
import threading

class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)


class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        # implement the efficient power of Q
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k



class myThread(threading.Thread):
    def __init__(self, num_q, p) -> None:
        super().__init__()
        self.num_q = num_q
        self.p = p

    def run(self) -> QuantumCircuit:
        print(self.num_q)
        A = BernoulliA(self.p)
        Q = BernoulliQ(self.p)
        problem = EstimationProblem(
            state_preparation=A,  # A operator
            grover_operator=Q,  # Q operator
            objective_qubits=[0],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
        )
        sampler = Sampler()
        ae = AmplitudeEstimation(
            num_eval_qubits=self.num_q,  # the number of evaluation qubits specifies circuit width and accuracy
            sampler=sampler,
        )
        qiskit_circ = ae.construct_circuit(problem)
        qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])
        num_q = qiskit_circ.num_qubits

        with open(f'qasm/qae_{num_q}.qasm', 'w') as f:
            f.write(qiskit_circ.qasm())
        # circuit : Circuit = Circuit.from_file(f'temp_qasm/{num_q * 2}.qasm')
        return qiskit_circ


if __name__ == '__main__':
    '''
    Based on paper to estimate quantum risk analysis. Essentially uses QAE, we can model with
    simple Bernoulli operator with Ry(p).
    Paper is here: 
    https://arxiv.org/pdf/1806.06893.pdf
    '''

    range_low = 4
    range_high = 20
    step = 2
    p = 0.3
    num_qubits = [x for x in range(range_low, range_high+1, step)]

    threads = [myThread(i, p) for i in num_qubits]

    qasm_list = [None] * len(num_qubits)

    for i, num_q in enumerate(num_qubits):
        qc = threads[i].start()
        qasm_list[i] = qc

    for t in threads:
        t.join()

    with open(f'finance_qae_list_{range_low}_{range_high}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)