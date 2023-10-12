import numpy as np
import random

def get_initial_population(config):
    """
        Populates variables from config and initialize P at gen 0.
        Parameters:
            config ( str ): path to config file
        Returns:
            g (int): current generation
            P (matrix or two D array): population of individuals
            W (int): Knapsack capacity
            S (list of tuples): Each tuple is an item (w_i , v_i)
            stop (int): final generation (stop condition)
    """
    seed_ = random.randint(0, 65536)
    print("Seed used:", seed_)
    np.random.seed(seed_)

    # Populate the problem varibles
    with open(config, 'r') as file:
        lines = file.readlines()

    pop_size, num_genes, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines [4:]]

    # Initialize population at generation 0
    g = 0
    P = np.random.randint(2, size=(pop_size, num_genes))
    
    return P, W, S, g, stop, num_genes

initial_population, MAX_WEIGHT, ITEMS, _, NUM_STEPS, NUM_GENES = get_initial_population("config_1.txt")

def get_item(index):
    return ITEMS[index]

NUM_ITEMS = len(ITEMS)
