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
NUM_MULTISTARTS = 32

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
        #print(f'  Template {tid+1}/{len(all_template_locations)}...')
        dist = try_template(template_gates, target_unitary)
        if dist < instantiation_threshold:
            print(f'   Template {tid} has distance {dist}')
            label = tid
            break
    return label

def fancy_cycle_through_templates(
    index: int,
    all_template_locations : list, 
    target_unitary : np.array,
    blocks_to_retry : list[int],
    instantiation_threshold : float = 2e-16
) -> int:
    if index not in blocks_to_retry:
        return -1
    print(f' Working on block: {index}')
    return cycle_through_templates(
        all_template_locations,
        target_unitary,
        instantiation_threshold
    )

def cycle_through_select_blocks(
    circuit : Circuit, 
    template_locations : list[tuple[int,int]],
    blocks_to_retry : list[int],
    instantiation_threshold : float = 2e-16
) -> list[tuple[int, np.array]]:

    num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        unitaries = [block.get_unitary().numpy for block in circuit]
        arguments = [
            (
                index,
                template_locations, 
                target_unitary, 
                blocks_to_retry,
                instantiation_threshold
            )
            for index, target_unitary in enumerate(unitaries) 
            if len(target_unitary) == 2**3
        ]
        labels = pool.starmap(fancy_cycle_through_templates, arguments)
    
    return [lu for lu in zip(labels, unitaries)]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('circuit_type')
    parser.add_argument('connectivity')
    args = parser.parse_args()

    instantiation_threshold = 2e-16
    base_name = f'{args.circuit_type}_{args.connectivity}'

    with open(f'partitioned-{args.circuit_type}_list_3_65.pickle', 'rb') as f:
        partitioned_circuits_list = pickle.load(f)
    with open(f'reinstantiate-{base_name}.pickle','rb') as f:
        block_nums_to_retry = pickle.load(f)
    with open(f'../templates_{args.connectivity}.pickle', 'rb') as f:
        template_locations = pickle.load(f)

    labels_and_unitaries : list[tuple[int,np.array]] = [] 
    
    for circ_num, circuit in enumerate(partitioned_circuits_list):
        output_name = f'{base_name}_{circ_num}.pickle'
        if exists(output_name):
            continue

        print(f'Circuit {circ_num+1}/{len(partitioned_circuits_list)}...')
        labels_and_unitaries = cycle_through_select_blocks(
            circuit, 
            template_locations,
            block_nums_to_retry[circ_num]
        )

        with open(output_name, 'wb') as f:
            pickle.dump(labels_and_unitaries, f)
