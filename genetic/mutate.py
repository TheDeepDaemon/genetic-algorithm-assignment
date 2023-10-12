import numpy as np

def mutate_genes(genes, mutation_freq):
    '''
        Randomly switch the value of some of
        the genes with the probability dictated by mutation_freq.
        This function assumes that the genes are boolean values.
    '''
    new_genes = np.copy(genes)
    for i in range(len(new_genes)):
        if np.random.random() < mutation_freq:
            new_genes[i] = not new_genes[i]
    return new_genes
