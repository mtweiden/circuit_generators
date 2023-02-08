import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from argparse import ArgumentParser
from os import listdir
from matplotlib.colors import BoundaryNorm 
from matplotlib.colors import Colormap

from bqskit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.ccp import CCPGate
from bqskit.passes.util.blockanalysis import BlockAnalysisPass

datasets_directory = f'../instantiations'
outlier_threshold = 0.10

def circuit_width_from_file_name(file_name : str) -> int:
    return int(re.findall(r'\d+', file_name)[0]) + 3

def get_labels(list_of_results : list[tuple[int,np.array]]) -> list[int]:
    return [result[0] for result in list_of_results]

def get_all_labels(all_results : list[list[tuple[int,np.array]]]) -> list[list[int]]:
    return [get_labels(list_of_results) for list_of_results in all_results]

def file_names_to_results(list_of_files : list[str]) -> list[list[tuple[int,np.array]]]:
    all_results = []
    for file_name in list_of_files:
        with open(f'{datasets_directory}/{file_name}', 'rb') as f:
            all_results.append(pickle.load(f))
    return all_results

def vectorize_label_counts(labels : list[int], num_templates : int) -> np.array:
    count_vector = np.zeros((num_templates,))
    for label in labels:
        count_vector[label] += 1
    return count_vector

def get_zero_rows(count_matrix : np.array) -> list[int]:
    return reversed(np.where(~count_matrix.any(axis=1))[0])

def used_labels(count_matrix : np.array, num_templates : int) -> list[int]:
    return [x for x in range(num_templates) if x not in get_zero_rows(count_matrix)]

def compress_count_matrix(count_matrix : np.array) -> np.array:
    zero_rows = get_zero_rows(count_matrix)
    for zero_row in zero_rows:
        count_matrix = np.delete(count_matrix, zero_row, axis=0)
    return count_matrix

def vectorize_all_label_counts(
    all_labels : list[int], 
    num_templates : int,
    normalize_columns : bool = True
) -> np.array: 
    num_circuits = len(all_labels)
    count_matrix = np.zeros((num_templates, num_circuits))
    for i, labels in enumerate(all_labels):
        count_vector = vectorize_label_counts(labels, num_templates)
        if normalize_columns:
            count_vector /= np.sum(count_vector)
        count_matrix[:,i] = count_vector
    return count_matrix

def num_elements_in_range(
    matrix : np.array,
    low_bound : float,
    high_bound : float,
) -> int:
    return np.sum(np.logical_and(matrix < high_bound, matrix >= low_bound))

def find_best_bins(
    matrix : np.array, 
    num_bins : int, 
    delta : float = 1e-4
) -> list[float]:
    num_nonzeros = np.count_nonzero(matrix)
    count_per_bin = num_nonzeros / num_bins
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    num_steps = int(np.ceil((max_val - min_val) / delta))

    lower_bounds = [min_val]

    current_lower = min_val + delta
    for x in range(1,num_steps+1):
        current_upper = delta * x + min_val
        count_in_range = num_elements_in_range(matrix, current_lower, current_upper)
        if count_in_range >= count_per_bin:
            lower_bounds.append(current_upper)
            current_lower = current_upper
    lower_bounds.append(max_val)
    
    return sorted(lower_bounds)

def show_2d_distribution(
    matrix : np.array,
    template_numbers : list[int],
    benchmark_widths : list[int],
) -> None:
    fig, ax = plt.subplots()
    num_bins = 10
    boundaries = find_best_bins(matrix, num_bins)
    norm = BoundaryNorm(boundaries, 256)

    im = ax.imshow(matrix, norm=norm, aspect='auto')
    cbar = fig.colorbar(im, ticks=boundaries, spacing='proportional')

    ax.set_xticks(range(len(benchmark_widths)))
    ax.set_yticks(range(len(template_numbers)))
    ax.set_xticklabels(benchmark_widths)
    ax.set_yticklabels(template_numbers)
    plt.show()

def matrix_to_vector(matrix: np.array) -> np.array:
    vector = np.sum(matrix, axis=1) 
    return vector / np.sum(vector)

def show_1d_distribution(
    vector : np.array,
    template_numbers : list[int],
) -> None:
    fig, ax = plt.subplots()
    ax.bar([x for x in range(len(vector))], vector)
    ax.set_xticks(range(len(template_numbers)))
    ax.set_xticklabels(template_numbers)
    #num_bins = 12
    #boundaries = find_best_bins(vector, num_bins)
    #norm = BoundaryNorm(boundaries, 256)

    #im = ax.imshow(np.expand_dims(vector, axis=1), norm=norm, aspect='auto')
    ##cbar = fig.colorbar(im, ticks=boundaries, spacing='proportional')
    #cbar = fig.colorbar(im, ticks=boundaries)

    #ax.set_yticks(range(len(template_numbers)))
    #ax.set_yticklabels(template_numbers)
    plt.show()

def labels_to_num_cnots(
    count_matrix : np.array,
    template_list : list[list[tuple[int,int]]],
    used_templates : list[int], 
) -> np.array:
    label_to_cnots = [len(t) for t in template_list]
    cnot_nums = {label:label_to_cnots[label] for label in used_templates}
    num_cnot_bins = max(label_to_cnots)

    _, cols = np.shape(count_matrix)
    cnot_matrix = np.zeros((num_cnot_bins+1, cols))

    for j in range(cols):
        for i, label in enumerate(used_templates):
            bin = label_to_cnots[label]
            val = count_matrix[i,j]
            cnot_matrix[bin, j] += val
    
    return cnot_matrix

def blocks_to_reinstantiate(
    all_results : list[list[tuple[int,np.array]]],
    label_use_vector : np.array,
    used_templates : list[int],
    usage_threshold : float = 0.5,
) -> list[list[int]]:
    labels_to_retry = []
    for usage, label in zip(label_use_vector, used_templates):
        if usage < usage_threshold:
            labels_to_retry.append(label)
    all_block_nums_to_retry = []
    for results in all_results:
        block_nums_to_retry = []
        for block_num, (label, _) in enumerate(results):
            if label in labels_to_retry:
                block_nums_to_retry.append(block_num)
        all_block_nums_to_retry.append(block_nums_to_retry)
    return all_block_nums_to_retry

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('circuit_type')
    parser.add_argument('connectivity')
    args = parser.parse_args()

    num_templates = 353 if args.connectivity == 'linear' else 8632

    dataset_name = f'{args.circuit_type}_{args.connectivity}'
    files_location = '../final_template_assignments'
    #files_location = '../instantiation'
    circuits = sorted(
        [x for x in listdir(files_location) if dataset_name in x],
        key=lambda x: circuit_width_from_file_name(x)
    )
    max_width = circuit_width_from_file_name(circuits[-1])
    all_results = file_names_to_results(circuits)
    all_labels = get_all_labels(all_results)

    # Distribution of labels by circuit width
    label_matrix = vectorize_all_label_counts(all_labels, num_templates)
    template_numbers = used_labels(label_matrix, num_templates)
    benchmark_widths = [x for x in range(3,max_width+1)]
    compressed_label_matrix = compress_count_matrix(label_matrix)

    show_2d_distribution(compressed_label_matrix, template_numbers, benchmark_widths)
    
    # Distribution of labels
    label_matrix = vectorize_all_label_counts(all_labels, num_templates, False)
    label_matrix = compress_count_matrix(label_matrix)
    label_vector = matrix_to_vector(label_matrix)

    show_1d_distribution(label_vector, template_numbers)

    # CNOTs after instantiation by circuit width
    with open(f'../templates_{args.connectivity}.pickle','rb') as f:
        template_list = pickle.load(f)
        cnots_numbers = list(set([len(t) for t in template_list]))
    label_matrix = vectorize_all_label_counts(all_labels, num_templates)
    cnot_matrix = labels_to_num_cnots(label_matrix, template_list, template_numbers)
    #show_2d_distribution(cnot_matrix, cnots_numbers, benchmark_widths)

    # CNOTs before instantiation by circuit width

    # Iterate through labeled blocks
    blocks_to_retry = blocks_to_reinstantiate(
        all_results, label_vector, template_numbers, outlier_threshold
    )
    with open(f'reinstantiate-{dataset_name}.pickle','wb') as f:
        pickle.dump(blocks_to_retry, f)
    with open(f'used_templates-{dataset_name}.pickle','wb') as f:
        pickle.dump(template_numbers, f)
        #used_templates = [template_list[t] for t in template_numbers]
        #pickle.dump(used_templates, f)
    # Make a list of all blocks that have more CNOTs than the median
    # Prepare those blocks for re-instantiation
