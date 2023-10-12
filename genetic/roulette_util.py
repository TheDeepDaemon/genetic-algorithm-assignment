import numpy as np


def convert_to_probabilities(population_evals: np.ndarray):
    '''
        Take an array of numbers,
        convert it to an array of probabilities
        that add up to 1.
    '''
    min_ = np.min(population_evals)
    
    # shift, so the lowest is zero
    adjusted = [x - min_ for x in population_evals]
    adjusted = np.array(adjusted, dtype=float)
    
    total = np.sum(adjusted)

    if total != 0:
        adjusted /= total

    return adjusted


def choose_weighted(weights):
    '''
        Return an index, with each index being
        returned with a probability given by
        its corresponding weight.
    '''
    rn = np.random.random()
    total = 0.0
    for i, w in enumerate(weights):
        total += w
        if rn < total:
            return i
    return -1


def generate_weights(population: list, eval_func):
    '''
        Generate a set of weights for this population.
        Assumes the population is a list of (index, member) pairs.
    '''
    vals = np.array([eval_func(member) for _, member in population], dtype=float)
    return convert_to_probabilities(vals)


def select_n_from_population(population: np.ndarray, n: int, evaluation_func):
    '''
        Select a subset of the population with a weighted selection process.
    '''

    # create a list of indices 0 to the size of the population.
    # this represents which members of the population have
    # not been selected.
    indices_left = list(range(len(population)))

    # create a list of pairs that includes the original indices
    indexed_pop = [(i, member) for i, member in enumerate(population)]

    # select n members to keep
    for _ in range(n):
        # select the members that are left
        current_population = [indexed_pop[i] for i in indices_left]

        # create the list of weights
        weights = generate_weights(current_population, evaluation_func)

        # weighted selection
        selected_index = choose_weighted(weights)

        # retrieve the original index of the selected member
        original_index, _  = current_population[selected_index]

        # remove the index from the list
        indices_left.remove(original_index)
    
    # remove the members corresponding to the indices left
    return np.delete(population, indices_left, axis=0)
