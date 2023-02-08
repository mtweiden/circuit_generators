import numpy as np
from bqskit import Circuit
from bqskit.passes import QSearchSynthesisPass
from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.ccp import CCPGate

if __name__ == '__main__':
    num_trails = 10
    rand_angles = [2*np.pi*np.random.random() for _ in range(num_trails)]

    for trial_num, rand_angle in enumerate(rand_angles):
        print(f'trail number {trial_num}')
        circuit = Circuit(2)
        circuit.append_gate(CPGate(), [0,1], [rand_angle])
        qsearch = QSearchSynthesisPass()
        qsearch.run(circuit)
        print(circuit.gate_counts)