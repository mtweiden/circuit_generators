# pylint: disable=line-too-long
from qiskit.algorithms import VQE
from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
# pylint: enable=line-too-long
import numpy as np
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit.opflow import TwoQubitReduction
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit_aer.noise import NoiseModel

import numpy as np
from qiskit.compiler.transpiler import transpile

import threading
import pickle

# backend = BasicAer.get_backend("qasm_simulator")

def get_qubit_op(atom: str):
    # Define Molecule
    driver = PySCFDriver(atom=atom, charge=0, spin=0)

    # Get properties
    properties = driver.run()
    num_particles = (properties
                        .get_property("ParticleNumber")
                        .num_particles)
    num_spin_orbitals = int(properties
                            .get_property("ParticleNumber")
                            .num_spin_orbitals)

    # Define Problem, Use freeze core approximation, remove orbitals.
    problem = ElectronicStructureProblem(driver)

    second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
    num_spin_orbitals = problem.num_spin_orbitals
    num_particles = problem.num_particles

    mapper = ParityMapper()  # Set Mapper
    hamiltonian = second_q_ops['ElectronicEnergy']  # Set Hamiltonian
    # Do two qubit reduction
    converter = QubitConverter(mapper,two_qubit_reduction=True)
    # reducer = TwoQubitReduction(num_particles)
    qubit_op = converter.convert(hamiltonian)
    # qubit_op = reducer.convert(qubit_op)

    return qubit_op, num_particles, num_spin_orbitals, problem, converter


def create_vqe(atom: str):
    (qubit_op, num_particles, num_spin_orbitals,
                             problem, converter) = get_qubit_op(atom)
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    var_form = UCCSD(converter,
                     num_particles,
                     num_spin_orbitals,
                     initial_state=init_state)
    vqe = VQE(var_form, SLSQP(maxiter=2))
    # vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    rand_parameters = np.random.rand(vqe.ansatz.num_parameters) * np.pi
    # rand_parameters = vqe_calc.optimal_parameters
    return vqe.construct_circuit(rand_parameters, qubit_op)


class myThread(threading.Thread):
    def __init__(self, atom: str, id: int) -> None:
        super().__init__()
        self.atom = atom
        self.id = id
        self.i = 0

    def run(self):
        circ = create_vqe(self.atom)[0]
        self.i = circ.num_qubits
        quantum_circuit = transpile(circ, basis_gates=['u3','cx'])
        with open(f'qasm/vqe_{self.id}_{self.i}.qasm', 'w') as f:
            f.write(quantum_circuit.qasm())
        return quantum_circuit


if __name__ == '__main__':
    # List of atoms to simulate
    # Need to add more
    atoms = [
        "Li .0 .0 .0; H .0 .0 -1.3",
        "Be .0 .0 .0; H .0 .0 -1.3; H .0 .0 1.3", 
        "B .0 .0 .0; H .0 .0 -1.3; H .0 .0 -.5; H .0 .0 1.3;",
        "C .0 .0 .0; H .0 .0 -1.3; H .0 .0 -.5; H .0 .0 1.3; H .0 .0 .3;"
    ]

    threads = [myThread(atom, i) for i,atom in enumerate(atoms)]

    qasm_list = [None] * len(atoms)

    for i, atom in enumerate(atoms):
        qc = threads[i].start()
        qasm_list[i] = qc

    # for atom in atoms:
    #     example = create_vqe(atom)
    #     circ = example[0]
    #     qiskit_circ = transpile(circ, basis_gates=['u3','cx'])
    #     print(qiskit_circ.qasm())

    actual_qubits = []
    for t in threads:
        t.join()
        actual_qubits.append(t.i)

    min_qubit = min(actual_qubits)
    max_qubit = max(actual_qubits)
    with open(f'vqa_list_{min_qubit}_{max_qubit}.pickle', 'wb') as f:
        pickle.dump(qasm_list, f)