import numpy as np
from genetic.genetic_algorithm import run_genetic_algorithm, sort_genes
from evaluation_function import evaluate_backpack
from get_item_values import get_total_weight, get_active_items
from load_config import initial_population, NUM_STEPS, NUM_GENES
from monte_carlo.monte_carlo_tree import search_space_mcts

def test_genetic_algorithm(
        num_trials,
        mutation_rate,
        n_to_select,
        goal_eval_value,
        selection_type):
    '''
        Uses the starter code provided in the
        assignment to test the genetic algorithm.
    '''

    final_values = []
    final_weights = []
    final_num_items = []

    for _ in range(num_trials):
        population = run_genetic_algorithm(
            initial_population=initial_population,
            num_steps=NUM_STEPS,
            evaluation_func=evaluate_backpack,
            mutation_freq=mutation_rate,
            n_to_select=n_to_select,
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

    print("________________\nGenetic Algorithm:\n")
    print("Selection type: {}".format(selection_type))
    print("Values: {} + - {} ".format(np.mean(final_values), np.std(final_values)))
    print()
    print("Weight: {} ".format(np.max(final_weights)))
    print()
    print("No of items: {} ".format(np.mean(final_no_items)))
    print()


def test_monte_carlo(num_trials, monte_carlo_iterations):
    '''
        Tests the monte carlo tree search based algorithm in order to compare results.
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


if __name__ == "__main__":
    num_trials = 30

    test_monte_carlo(
        num_trials=num_trials,
        monte_carlo_iterations=100
        )

    test_genetic_algorithm(
        num_trials=num_trials,
        mutation_rate=0.2,
        n_to_select=int(len(initial_population)*0.1),
        goal_eval_value=None, selection_type='tournament'
        )
