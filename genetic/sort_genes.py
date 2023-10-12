import numpy as np

def sort_genes(genes_list: np.ndarray, eval_func):
    '''
        Sort a list of genes based on the output of the evaluation function.
    '''

    evaluations = np.zeros(len(genes_list), dtype=float)

    for i, genes in enumerate(genes_list):
        evaluations[i] = eval_func(genes)
    
    indices = np.argsort(evaluations)

    return genes_list[indices]
