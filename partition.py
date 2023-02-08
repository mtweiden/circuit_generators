from bqskit import Circuit 
from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes.partitioning.scan import ScanPartitioner
import pickle
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    args = parser.parse_args()

    with open(args.file_name, 'rb') as f:
        circuit_list = pickle.load(f)
    
    new_circuit_list = []
    for i,circuit in enumerate(circuit_list):
        print(f'{i+1}/{len(circuit_list)}...')
        task = CompilationTask(circuit, [ScanPartitioner(3)])
        start_time = time.time()
        with Compiler() as compiler:
            partitioned_circuit = compiler.compile(task)
        new_circuit_list.append(partitioned_circuit)
        print(f'Time elapsed: {time.time()-start_time}s')
    
    with open(f'partitioned-{args.file_name}', 'wb') as f:
        pickle.dump(new_circuit_list, f)
    