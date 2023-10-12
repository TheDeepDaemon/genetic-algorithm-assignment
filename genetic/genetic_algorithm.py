import numpy as np
from genetic.combine_genes import one_point_crossover, two_point_crossover
from genetic.sort_genes import sort_genes
from genetic.mutate import mutate_genes
from genetic.roulette_util import select_n_from_population
from genetic.tournament_util import tournament_selection


def init_population(population_size, num_genes):
    '''
        Initialize the genes as a matrix of boolean values.
    '''
    arr = np.random.randint(0, 2, size=(population_size * num_genes))
    return np.reshape(arr, newshape=(population_size, num_genes))


def combine(parent1, parent2, num_genes):
    '''
        Combine according to one-point or two-point crossover.
    '''

    # choose the combination type
    # (restricted to single-point crossover right now
    # because of assignment specifications,
    # but normally would be random)
    comb_type = 0

    # call one of the functions based on the chosen type
    if comb_type == 0:
        return one_point_crossover(parent1, parent2, np.random.randint(0, num_genes))
    else:
        return two_point_crossover(parent1, parent2, np.random.randint(0, num_genes), np.random.randint(0, num_genes))


def regenerate_pop_from_subset(subset, population_size, num_genes, mutation_freq):
    '''
        Use a subset to regenerate the rest of the population.
    '''

    # initialize the new population with the chosen subset
    new_population = [member for member in subset]

    # fill in the rest of the population with new members
    for _ in range(population_size - len(subset)):
        parents = subset[np.random.choice(subset.shape[0], 2, replace=False), :]
        new_member = combine(parents[0], parents[1], num_genes)
        new_population.append(mutate_genes(new_member, mutation_freq))

    # make it a matrix again
    new_population = np.array(new_population)

    return new_population


def ga_step(population, evaluation_func, mutation_freq, n_to_select, selection_type):
    '''
        Run the genetic algorithm for one step.
        The subset is used as parents to fill in the rest of the population.
    '''
    
    # get sizes
    population_size = population.shape[0]
    num_genes = population.shape[1]

    # select the parents
    selected_members = None
    if selection_type == 'roulette':
        # This is roulette wheel selection, where a randomly chosen (weighted) subset is chosen.
        # The rest are eliminated.
        selected_members = select_n_from_population(population, n_to_select, evaluation_func)
    elif selection_type == 'tournament':
        # This is tournament selection, where the population is
        # divided up into groups, and the best is picked from each one.
        selected_members = tournament_selection(population, n_to_select, evaluation_func)
    else:
        # This is truncation selection, where the worst members are eliminated,
        # and the rest are used as parents.
        selected_members = sort_genes(population, evaluation_func)[-n_to_select:]

    # use the selected members as parents to regenerate the population
    # back to it's previous size
    new_population = regenerate_pop_from_subset(selected_members, population_size, num_genes, mutation_freq)

    return new_population


def ga_step_truncation2(population, evaluation_func, mutation_freq, num_new_members):
    '''
        Run the genetic algorithm for one step.
        This is truncation selection 2 (my own formulation),
        where new members are added and the best are kept.
    '''

    # get sizes
    population_size = population.shape[0]
    num_genes = population.shape[1]

    # init with existing population
    new_population = [member for member in population]

    # add new members
    for _ in range(num_new_members):
        parents = population[np.random.choice(population.shape[0], 2, replace=False), :]
        new_member = combine(parents[0], parents[1], num_genes)
        new_population.append(mutate_genes(new_member, mutation_freq))
    
    # make it a matrix again
    new_population = np.array(new_population)
    
    # keep the best members
    return sort_genes(new_population, evaluation_func)[-population_size:]


def get_best_population_evaluation(population, evaluation_func):
    '''
        Get the best evaluation value in the population.
    '''

    # init at negative infinity so
    # any value should be higher
    best_value = float("-inf")

    # iterate through every member in the population
    for member in population:

        # evaluate the member
        val = evaluation_func(member)

        # update if it is greater than the best value
        if val > best_value:
            best_value = val
    return best_value


def run_genetic_algorithm(
        initial_population,
        num_steps,
        evaluation_func,
        mutation_freq,
        n_to_select,
        selection_type,
        goal_eval_value=None):
    '''
        Run the genetic algorithm for a pre-determined number of steps.
        Stop if you have reached the goal evaluation value.
    '''
    
    # initialize the population
    population = initial_population

    for _ in range(num_steps):

        # update the population for this step
        population = ga_step(
            population=population,
            evaluation_func=evaluation_func,
            mutation_freq=mutation_freq,
            n_to_select=n_to_select,
            selection_type=selection_type)
        
        # extra stop condition
        if goal_eval_value is not None:

            # if the best value in the population is above goal_eval_value,
            # then stop and return the current population
            best_value = get_best_population_evaluation(population, evaluation_func)
            if best_value >= goal_eval_value:
                break
    return population
