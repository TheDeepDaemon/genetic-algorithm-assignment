import numpy as np
from genetic.sort_genes import sort_genes

def get_subgroups(population: np.ndarray, n_subgroups: int):
    '''
        Seperate the population out into sub-groups.
        n_subgroups determines how many groups there are.
    '''

    pop_size = len(population)

    subgroup_size = int(pop_size / n_subgroups)

    subgroups = []

    # iterate through sections of the population
    # of size=subgroup_size, and add to a subgroup.
    # a list of subgroups (tournaments) is created.
    for i in range(n_subgroups):
        low = i * subgroup_size
        high = (i + 1) * subgroup_size

        cur_subgroup = []

        for j in range(low, high):
            cur_subgroup.append(population[j])

        subgroups.append(cur_subgroup)
    
    # there will probably be some that haven't been added to a subgroup yet
    # add each to one of the existing subgroups
    last_element = n_subgroups * subgroup_size
    for i, j in enumerate(range(last_element, pop_size)):
        subgroups[i].append(population[j])
    
    return [np.array(subgroup) for subgroup in subgroups]


def tournament_selection(population: np.ndarray, n_to_select: int, eval_func):
    '''
        Divide the population into subgroups and select the best from each one.
    '''

    # shuffle the population so that how the groups are divided is random
    np.random.shuffle(population)

    # get the list of subgroups
    subgroups = get_subgroups(population, n_to_select)

    selected_members = []

    # iterate through each subgroup,
    # pick the highest fitness member of each subgroup
    for subgroup in subgroups:

        # sort according to fitness
        sorted_group = sort_genes(subgroup, eval_func)

        # select the top member
        selected_members.append(sorted_group[-1])
    
    return np.array(selected_members)
