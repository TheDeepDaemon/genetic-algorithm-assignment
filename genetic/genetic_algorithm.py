import numpy as np
from genetic.combine_genes import one_point_crossover, two_point_crossover
from genetic.sort_genes import sort_genes
from genetic.mutate import mutate_genes
from genetic.roulette_util import select_n_from_population
from genetic.tournament_util import tournament_selection
import matplotlib.pyplot as plt

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


def copy_parents(parents, population_size):
    '''
        Regenerate the population to it's original size
        by copying the parents.
        This is used to answer Q2, because selection
        is to be used without crossover or mutation.
    '''
    new_population = [p for p in parents]

    for _ in range(population_size - len(parents)):
        child =  parents[np.random.choice(len(parents), 1, replace=False), :]
        new_population.append(child[0])

    return np.array(new_population)


def ga_step(population, evaluation_func, mutation_freq, n_to_select, selection_type, selection_only=False):
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
    if selection_only:
        # just copy the parents
        new_population = copy_parents(selected_members, population_size=population_size)
    else:
        # use crossover and mutation
        new_population = regenerate_pop_from_subset(selected_members, population_size, num_genes, mutation_freq)

    return new_population


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


def get_average_evaluation(population, evaluation_func):
    '''
        Get the average evaluation value in the population.
    '''

    total = 0

    # iterate through every member in the population
    for member in population:

        # evaluate the member
        val = evaluation_func(member)
        total += val
    return total / len(population)


def run_genetic_algorithm(
        initial_population,
        num_generations,
        evaluation_func,
        mutation_freq,
        n_to_select,
        selection_type,
        goal_eval_value=None,
        selection_only=False):
    '''
        Run the genetic algorithm for a pre-determined number of steps.
        Stop if you have reached the goal evaluation value.
    '''
    
    # initialize the population
    population = initial_population

    for _ in range(num_generations):

        # update the population for this step
        population = ga_step(
            population=population,
            evaluation_func=evaluation_func,
            mutation_freq=mutation_freq,
            n_to_select=n_to_select,
            selection_type=selection_type,
            selection_only=selection_only)
        
        # extra stop condition
        if goal_eval_value is not None:

            # if the best value in the population is above goal_eval_value,
            # then stop and return the current population
            best_value = get_best_population_evaluation(population, evaluation_func)
            if best_value >= goal_eval_value:
                break
    return population


def graph_genetic_algorithm(
        initial_population,
        num_generations,
        evaluation_func,
        mutation_freq,
        n_to_select,
        selection_type,
        selection_only):
    '''
        Run the genetic algorithm.
        This version of the function is designed 
        to be run in order to graph results for each generation.
    '''
    
    # initialize the population
    population = initial_population

    best_fitness_log = []
    avg_fitness_log = []

    for _ in range(num_generations):

        # update the population for this step
        population = ga_step(
            population=population,
            evaluation_func=evaluation_func,
            mutation_freq=mutation_freq,
            n_to_select=n_to_select,
            selection_type=selection_type,
            selection_only=selection_only)
        
        best_fitness = get_best_population_evaluation(
            population=population,
            evaluation_func=evaluation_func)
        best_fitness_log.append(best_fitness)
        
        avg_fitness = get_average_evaluation(
            population=population, 
            evaluation_func=evaluation_func)
        avg_fitness_log.append(avg_fitness)
    
    plt.plot(best_fitness_log)
    plt.plot(avg_fitness_log)

    plt.savefig('fitness plot.png', bbox_inches='tight')
