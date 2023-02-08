import pickle
from argparse import ArgumentParser

from bqskit import Circuit
from bqskit.passes.util.blockanalysis import BlockAnalysisPass
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.ccp import CCPGate

def gate_counts_by_block(circuit : Circuit) -> list[dict]:
    """
    Return a list of dictionaries, where each dictionary corresponds to the
    gates used in some block.
    """
    filter = lambda x: x.num_qudits == 3
    analysis = BlockAnalysisPass(filter_function=filter)
    analysis.run(circuit)
    return analysis.results

def convert_to_cnots(gate_counts : list[dict[Gate,int]]) -> list[int]:
    # Conversion factors: 
    #   CCP -> 6 CNOTs
    #   CP  -> 2 CNOTs
    cnots = gate_counts[CNOTGate()]
    ccps  = gate_counts[CCPGate()]
    cps   = gate_counts[CPGate()]
    return cnots + 6*ccps + 2*cps

def cnots_in_template(label : int, templates : list[tuple[int,int]]) -> int:
    return len(templates[label])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('circuit_list')
    parser.add_argument('instantiations')
    args = parser.parse_args()

    with open(args.circuit_list, 'rb') as f:
        circuit_list = pickle.load(f)
    with open(args.instantiations, 'rb') as f:
        instantiations = pickle.load(f)
    
    graph_type = 'linear' if 'linear' in instantiations else 'complete'
    with open(f'../templates_{graph_type}.pickle','rb') as f:
        templates = pickle.load(f)
    
    results = []
    for circuit, labels_and_unitaries in zip(circuit_list, instantiations):
        # original
        gate_counts = gate_counts_by_block(circuit)
        original_cnots = [convert_to_cnots(gc) for gc in gate_counts]

        labels = [l for (l,u) in labels_and_unitaries]
        template_cnots = [cnots_in_template(l,templates) for l in labels]

        original_to_template = [o/t for o,t in zip(original_cnots, template_cnots)]
        results.append(results)
    print(results)
