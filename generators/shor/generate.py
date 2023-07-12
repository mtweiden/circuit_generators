"""Based off the algorithm in https://arxiv.org/pdf/quant-ph/0205095"""
from bqskit import Circuit
from bqskit.ir.gates.parameterized.u1 import U1Gate 
from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.ccp import CCPGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.ccx import ToffoliGate
from bqskit.ir.lang.qasm2 import OPENQASM2Language
import numpy as np
from os.path import exists
from qiskit.circuit.library.basis_change.qft import QFT
import pickle


# Phi ADD(a)
def subcircuit_add_x(x : int, n : int, inverse : bool = False) -> Circuit:
    """
    See Figure 3 of referenced paper. Value of a is hardcoded in single qubit 
    gate rotations.
    """
    circuit = Circuit(n)

    for i in range(n):
        angles = []
        for k, j in enumerate(range(i, n)):
            if x & (1 << j):
                angles.append(2 ** (k+1))
        ang = (-1)**inverse * np.sum(2 * np.pi / np.array(angles))
        circuit.append_gate(U1Gate(), [i], [ang])
    
    return circuit

def subcircuit_cadd_x(x : int, n : int, inverse : bool = False) -> Circuit:
    """
    See Figure 3 of referenced paper. Value of a is hardcoded in single qubit 
    gate rotations.
    """
    circuit = Circuit(n + 1)

    for i in range(n):
        angles = []
        for k, j in enumerate(range(i, n)):
            if x & (1 << j):
                angles.append(2 ** (k+1))
        ang = (-1)**inverse * np.sum(2 * np.pi / np.array(angles))
        # offset i by 1 so control goes at the top
        circuit.append_gate(CPGate(), [0, i+1], [ang])
    
    return circuit

def subcircuit_ccadd_x(x : int, n : int, inverse : bool = False) -> Circuit:
    """
    See Figure 3 of referenced paper. Value of a is hardcoded in single qubit 
    gate rotations.
    """
    circuit = Circuit(n + 2)

    for i in range(n):
        angles = []
        for k, j in enumerate(range(i, n)):
            if x & (1 << j):
                angles.append(2 ** (k+1))
        ang = (-1)**inverse * np.sum(2 * np.pi / np.array(angles))
        # offset i by 2 so controls go at the top
        circuit.append_gate(CCPGate(), [0, 1, i+2], [ang])
    
    return circuit

# Phi ADD(N) gate (where we've hardcoded N)

# QFT 
def subcircuit_qft(n : int, inverse = False) -> Circuit:
    qasm_file_name = f'qasm/'
    qasm_file_name += f'qft_{n}.qasm' if not inverse else f'iqft_{n}.qasm'
    if not exists(qasm_file_name):
        qiskit_circ = QFT(num_qubits=n, inverse=inverse)
        #qiskit_circ = transpile(qiskit_circ, basis_gates=['u3','cx'])
        with open(qasm_file_name, 'w') as f:
            f.write(qiskit_circ.qasm())
    return Circuit.from_file(qasm_file_name)

# Phi ADD(a) mod N 
def subcircuit_add_a_mod_N(n : int, a : int, N : int, inverse : bool = False) -> Circuit:
    """
    See Figure 5 in reference paper.

    Returns circuit with n+3 qubits
    [0-1]   - control qubits
    [2]     - ancilla qubit
    [3-n+2] - b register
    """
    add_N = subcircuit_add_x(a, n)
    cadd_N = subcircuit_cadd_x(N, n)
    ccadd_a = subcircuit_ccadd_x(a, n)
    iadd_N = subcircuit_add_x(a, n, inverse=True)
    icadd_N = subcircuit_cadd_x(N, n, inverse=True)
    iccadd_a = subcircuit_ccadd_x(a, n, inverse=True)
    qft = subcircuit_qft(n)
    iqft = subcircuit_qft(n, inverse=True)

    c1, c2, ancilla = 0, 1, 2
    # 3..n+3 - b register
    b_reg = [_ for _ in range(3, n+3)]
    loc_c1_c2_b = [c1, c2] + b_reg
    loc_ancilla_b = [ancilla] + b_reg
    loc_bmsb_aniclla = [n+2, ancilla]
    loc_bmsb = [n+2]

    circuit = Circuit(n + 3)
    if not inverse:
        circuit.append_circuit(ccadd_a, loc_c1_c2_b)
        circuit.append_circuit(iadd_N, b_reg)
        circuit.append_circuit(iqft, b_reg)
        circuit.append_gate(CNOTGate(), loc_bmsb_aniclla)
        circuit.append_circuit(qft, b_reg)
        circuit.append_circuit(cadd_N, loc_ancilla_b)
        circuit.append_circuit(iccadd_a, loc_c1_c2_b)
        circuit.append_circuit(iqft, b_reg)
        circuit.append_gate(XGate(), loc_bmsb)
        circuit.append_gate(CNOTGate(), loc_bmsb_aniclla)
        circuit.append_gate(XGate(), loc_bmsb)
        circuit.append_circuit(qft, b_reg)
        circuit.append_circuit(ccadd_a, loc_c1_c2_b)
    else:
        circuit.append_circuit(iccadd_a, loc_c1_c2_b)
        circuit.append_circuit(iqft, b_reg)
        circuit.append_gate(XGate(), loc_bmsb)
        circuit.append_gate(CNOTGate(), loc_bmsb_aniclla)
        circuit.append_gate(XGate(), loc_bmsb)
        circuit.append_circuit(qft, b_reg)
        circuit.append_circuit(ccadd_a, loc_c1_c2_b)
        circuit.append_circuit(icadd_N, loc_ancilla_b)
        circuit.append_circuit(iqft, b_reg)
        circuit.append_gate(CNOTGate(), loc_bmsb_aniclla)
        circuit.append_circuit(qft, b_reg)
        circuit.append_circuit(add_N, b_reg)
        circuit.append_circuit(iccadd_a, loc_c1_c2_b)

    return circuit


# CMULT(a) mod N
def subcircuit_cmult_a_mod_N(n : int, a : int, N : int, inverse : bool = False) -> Circuit:
    """
    Figure 6 in reference.
    
    [0]        - control qubit
    [1]        - ancilla qubit
    [2-n+1]    - x register
    [n+2-2n+1] - b register
    """
    qft = subcircuit_qft(n)
    iqft = subcircuit_qft(n, inverse=True)
    ccadds = [subcircuit_add_a_mod_N(n, a * 2**i, N, inverse=inverse) for i in range(n)]

    control = [0]
    ancilla = [1]
    x_reg = [_ for _ in range(2, n+2)]
    b_reg = [_ for _ in range(n+2, 2*n+2)]

    circuit = Circuit(2*n + 2)
    if not inverse:
        circuit.append_circuit(qft, b_reg)
        for i, ccadd in enumerate(ccadds):
            location = control + [i+2] + ancilla + b_reg
            circuit.append_circuit(ccadd, location)
        circuit.append_circuit(iqft, b_reg)
    else:
        circuit.append_circuit(qft, b_reg)
        for i, ccadd in enumerate(reversed(ccadds)):
            j = n - i - 1
            location = control + [j+2] + ancilla + b_reg
            circuit.append_circuit(ccadd, location)
        circuit.append_circuit(iqft, b_reg)

    return circuit

# CONTROLLED SWAP
def subcircuit_controlled_swap(n : int) -> Circuit:
    """
    Figure 10 in reference.
    
    [0]        - control
    [1-n]    - register 1
    [n+1-2n] - register 2
    """
    circuit = Circuit(2*n + 1)
    cnot_locations = [[i+1,n+i+1] for i in range(n)]

    for location in cnot_locations:
        circuit.append_gate(CNOTGate(), location)
        circuit.append_gate(ToffoliGate(), [0] + location)
        circuit.append_gate(CNOTGate(), location)
    
    return circuit

# Modular exponentiation
def modular_exponentiation(n : int, a : int, N : int) -> Circuit:
    """
    Figure 7 in reference.

    [0]        - control
    [1]        - ancilla
    [2-n+1]    - x register
    [n+2-2n+1] - b register
    """
    circuit = Circuit(2*n+2)
    cmult = subcircuit_cmult_a_mod_N(n, a, N)
    icmult = subcircuit_cmult_a_mod_N(n, a, N, inverse=True)
    cswap = subcircuit_controlled_swap(n)

    control = [0]
    ancilla = [1]
    x_reg = [_ for _ in range(2,n+2)]
    b_reg = [_ for _ in range(n+2,2*n+2)]

    circuit.append_circuit(cmult, control + ancilla + x_reg + b_reg)
    circuit.append_circuit(cswap, control + x_reg + b_reg)
    circuit.append_circuit(icmult, control + ancilla + x_reg + b_reg)

    return circuit


def format_arguments(N: int) -> tuple[int,int,int]:
    """
    Given a value for N, return a tuple containing:
        (n, a, N)
    n - number of bits in N
    a - a value between 1 and N-1 (just returns N//2)
    N - the value of N
    """
    return (int(np.ceil(np.log2(N))), N//2, N)

if __name__ == '__main__':
    #n = 4
    #a = 7
    #N = 15
    #circ = modular_exponentiation(n,a,N)
    #circ.unfold_all()
    #qasm = circ.to('qasm')
    #print(qasm)

    # (n, a, N)
    # n - bits in a and N
    # a - N // 2
    # N - prime 2
    with open('values_for_N.pickle', 'rb') as f:
        values_for_N = pickle.load(f)

    arguments = [format_arguments(N) for N in values_for_N]

    circuits = []
    for i,args in enumerate(arguments):
        print(f'Generating circuit {i+1}/{len(arguments)}...')
        circ = modular_exponentiation(*args)
        circ.unfold_all()
        num_q = circ.num_qudits
        with open(f'qasm/shor_{num_q}.qasm', 'w') as f:
            f.write(OPENQASM2Language().encode(circuit=circ))
        with open(f'pickle/shor_{num_q}.pkl', 'wb') as pf:
            pickle.dump(circ, pf)
        circuits.append(circ)

    min_size = circuits[0].num_qudits
    max_size = circuits[-1].num_qudits

    with open(f'shor_list_{min_size}_{max_size}.pickle','wb')  as f:
        pickle.dump(circuits, f)
