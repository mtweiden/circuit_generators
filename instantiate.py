from bqskit import Circuit
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.constant.cx import CNOTGate
from argparse import ArgumentParser
import pickle
import numpy as np
from os.path import exists
from multiprocessing import Pool, cpu_count

WRONG_SIZE  = -2
INST_FAILED = -1
NUM_MULTISTARTS = 10

def no_error(label : int) -> int:
    return label >= 0

def init_circuit(circuit : Circuit) -> Circuit:
    for q in range(circuit.num_qudits):
        circuit.append_gate(U3Gate(), [q])
    return circuit

def insert_atom(circuit : Circuit, location : tuple[int]) -> Circuit:
    circuit.append_gate(CNOTGate(), location)
    circuit.append_gate(U3Gate(), [location[0]])
    circuit.append_gate(U3Gate(), [location[1]])
    return circuit

def get_template(gate_locations : list[tuple[int]], num_q : int = 3) -> Circuit:
    circuit = init_circuit(Circuit(num_q))
    for location in gate_locations:
        circuit = insert_atom(circuit, location)
    return circuit

def try_template(template_gates : Circuit, target_unitary : np.array) -> float:
    template = get_template(template_gates)
    template.instantiate(target_unitary, multistarts=NUM_MULTISTARTS)
    return template.get_unitary().get_distance_from(target_unitary)

def cycle_through_templates(
    all_template_locations : list, 
    target_unitary : np.array,
    instantiation_threshold : float = 2e-16
) -> int:
    label = INST_FAILED
    for tid, template_gates in enumerate(all_template_locations):
        print(f'  Template {tid+1}/{len(all_template_locations)}...')
        dist = try_template(template_gates, target_unitary)
        if dist < instantiation_threshold:
            print(f'   Template {tid} has distance {dist}')
            label = tid
            break
    return label

def cycle_through_blocks(
        circuit : Circuit, 
        template_locations : list[tuple[int,int]],
        instantiation_threshold : float = 2e-16
) -> list[tuple[int, np.array]]:

    num_processes = cpu_count() // 2

    with Pool(processes=num_processes) as pool:
        unitaries = [block.get_unitary().numpy for block in circuit]
        arguments = [
            (template_locations, target_unitary, instantiation_threshold)
            for target_unitary in unitaries if len(target_unitary) == 2**3
        ]
        labels = pool.starmap(cycle_through_templates, arguments)
    
    return [lu for lu in zip(labels, unitaries)]

    # these_labels_and_unitaries = []
    #for block_num, block in enumerate(circuit):
    #    if 2**block.num_qudits != 2**3:
    #        continue
    #    print(f' Block {block_num+1}/{len(circuit)}...')
    #    target_unitary = block.get_unitary().numpy
    #    label = cycle_through_templates(
    #        template_locations, 
    #        target_unitary, 
    #        instantiation_threshold
    #    )
    #    these_labels_and_unitaries += [(label, target_unitary)]
    #return these_labels_and_unitaries

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('graph_type')
    args = parser.parse_args()

    instantiation_threshold = 2e-16

    out_names = ['qft', 'tfim']
    out_name = 'unknown'
    for x in out_names:
        if x in args.file_name:
            out_name = x
            break

    with open(args.file_name, 'rb') as f:
        partitioned_circuits_list = pickle.load(f)
    with open(f'templates_{args.graph_type}.pickle', 'rb') as f:
        template_locations = pickle.load(f)

    labels_and_unitaries : list[tuple[int,np.array]] = [] 
    
    for circ_num, circuit in enumerate(partitioned_circuits_list):
        output_name = f'{out_name}_{args.graph_type}_{circ_num}.pickle'
        if exists(output_name):
            continue

        print(f'Circuit {circ_num+1}/{len(partitioned_circuits_list)}...')
        labels_and_unitaries = cycle_through_blocks(circuit, template_locations)

        with open(output_name, 'wb') as f:
            pickle.dump(labels_and_unitaries, f)
