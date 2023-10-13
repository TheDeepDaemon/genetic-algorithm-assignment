import numpy as np
from genetic.genetic_algorithm import run_genetic_algorithm, graph_genetic_algorithm, sort_genes
from evaluation_function import evaluate_backpack
from get_item_values import get_total_weight, get_active_items
from load_config import NUM_STEPS, NUM_GENES, CONFIG_FNAME, POP_SIZE
from monte_carlo.monte_carlo_tree import search_space_mcts
import pandas as pd
import os

def generate_initial_population(population_size):
    return np.random.randint(2, size=(population_size, NUM_GENES))

def test_genetic_algorithm(
        num_trials,
        mutation_rate,
        percent_parents,
        goal_eval_value,
        selection_type,
        population_sizes):
    '''
        Uses the starter code provided in the
        assignment to test the genetic algorithm.
    '''

    selection_type_log = []
    value_log = []
    max_weight_log = []
    n_items_log = []

    for pop_size in population_sizes:

        initial_population = generate_initial_population(population_size=pop_size)

        final_values = []
        final_weights = []
        final_num_items = []

        for _ in range(num_trials):
            population = run_genetic_algorithm(
                initial_population=initial_population,
                num_generations=NUM_STEPS,
                evaluation_func=evaluate_backpack,
                mutation_freq=mutation_rate,
                n_to_select=int(percent_parents * pop_size),
                selection_type=selection_type,
                goal_eval_value=goal_eval_value)
            
            best_chromosome = sort_genes(population, evaluate_backpack)[-1]

            best_fitness = evaluate_backpack(best_chromosome)
            best_weight = get_total_weight(best_chromosome)
            num_items = get_active_items(best_chromosome)

            final_values.append(best_fitness)
            final_weights.append(best_weight)
            final_num_items.append(num_items)

        final_values = np.array(final_values)
        final_weights = np.array(final_weights)
        final_no_items = np.array(final_num_items)

        selection_type_log.append(selection_type)
        value_log.append("{} + - {}".format(np.mean(final_values), np.std(final_values)))
        max_weight_log.append(np.max(final_weights))
        n_items_log.append(np.mean(final_no_items))
    
    data = {
        'Population size': population_sizes,
        'Selection type': selection_type,
        'Knapsack Total Value': value_log,
        'Knapsack Max Weight': max_weight_log,
        '# of items in Best Solution': n_items_log}
    
    output_fname = os.path.splitext(CONFIG_FNAME)[0]

    df = pd.DataFrame(data=data)

    if os.path.exists(output_fname + ".csv"):
        df.to_csv(output_fname + '.csv', index=False, mode='a', header=False)
    else:
        df.to_csv(output_fname + '.csv', index=False)


def track_generational_performance(
        mutation_rate,
        percent_parents,
        selection_type,
        population_size,
        selection_only):
    '''
        Runs the genetic algorithm and displays
        the fitness for each generation as a plot.
    '''
    
    population = generate_initial_population(population_size=population_size)

    graph_genetic_algorithm(
        population,
        NUM_STEPS,
        evaluate_backpack,
        mutation_rate,
        int(percent_parents * population_size),
        selection_type,
        selection_only)


def test_monte_carlo(num_trials, monte_carlo_iterations):
    '''
        Tests the monte carlo tree search based algorithm in order to compare results.
        Testing code is based on the code given in the assignment.
        This is to answer "Problem 2: Compare GA with a Non-population-based Search Algorithm"
    '''

    initial_state = np.zeros(NUM_GENES, dtype=bool)

    final_values = []
    final_weights = []
    final_num_items = []

    for _ in range(num_trials):
        result = search_space_mcts(
            initial_state=initial_state,
            eval_func=evaluate_backpack,
            num_decisions=NUM_STEPS,
            monte_carlo_iterations=monte_carlo_iterations)
        
        best_fitness = evaluate_backpack(result)
        best_weight = get_total_weight(result)
        num_items = get_active_items(result)

        final_values.append(best_fitness)
        final_weights.append(best_weight)
        final_num_items.append(num_items)
    
    final_values = np.array(final_values)
    final_weights = np.array(final_weights)
    final_no_items = np.array(final_num_items)

    print("________________\nMonte Carlo:\n")
    print("Values: {} + - {} ".format(np.mean(final_values), np.std(final_values)))
    print()
    print("Weight: {} ".format(np.max(final_weights)))
    print()
    print("No of items: {} ".format(np.mean(final_no_items)))
    print()


def q2():
    '''
        Code used to answer question 2.
    '''
    track_generational_performance(0.2, 0.2, 'tournament', POP_SIZE, True)


def q3():
    '''
        Code used to answer question 3.
    '''
    track_generational_performance(0.2, 0.2, 'tournament', POP_SIZE, False)


def q4():
    '''
        Code used to answer question 4.
    '''

    num_trials = 30

    population_sizes = [POP_SIZE]

    test_genetic_algorithm(
        num_trials=num_trials,
        mutation_rate=0.2,
        percent_parents=0.2,
        goal_eval_value=None,
        selection_type='tournament',
        population_sizes=population_sizes
        )


def q5():
    '''
        Code used to answer problem 2, or question 5.
    '''

    num_trials = 30
    monte_carlo_iterations = 100

    test_monte_carlo(num_trials, monte_carlo_iterations)


if __name__ == "__main__":
    q3()
