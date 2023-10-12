from monte_carlo.monte_carlo_node import Node


class MonteCarloTree:
    
    def __init__(self, initial_state, eval_func) -> None:
        '''
            Create tree.
        '''
        self.root = Node(None, initial_state)
        self.eval_func = eval_func
    
    def mcts_step(self):
        '''
            Single step of monte carlo tree search.
        '''
        selected_node = self.root

        # select a node that hasn't been expanded
        while selected_node.children is not None:
            selected_node = selected_node.get_highest_ucb1_child()

        # expand that node
        selected_node.expand()

        # rollout
        rollout_result = \
            selected_node.rollout(self.eval_func)

        # backprop
        selected_node.pass_result_back(rollout_result)
    
    def monte_carlo_tree_search(self, iterations):
        '''
            Run monte carlo tree search.
        '''
        for _ in range(iterations):
            self.mcts_step()
        
        return self.root.child_with_most_sims()
    
    def show_tree(self):
        '''
            Display tree to console.
        '''
        current_nodes = [self.root]
        while len(current_nodes) > 0:
            next_nodes = []
            print("\n::  ", end='')
            for node in current_nodes:
                print("({}, {})".format(node.genome, self.eval_func(node.genome)), end=' ')
                if node.children is not None:
                    for child in node.children:
                        next_nodes.append(child)
            current_nodes = next_nodes
            print()
    
    def make_decision(self, index):
        '''
            Make a decision using the index of the child.
        '''
        self.root = self.root.children[index]
        self.root.parent = None


def search_space_mcts(initial_state, eval_func, num_decisions, monte_carlo_iterations):
    '''
        Use monte carlo tree search to search the space of possible solutions.
        This is the algorithm being compared to GA
        to answer "Problem 2: Compare GA with a Non-population-based Search Algorithm"
    '''
    tree = MonteCarloTree(initial_state, eval_func)

    best_genome = None
    best_eval = float('-inf')

    for _ in range(num_decisions):

        # get the MCTS decision
        result_index = tree.monte_carlo_tree_search(monte_carlo_iterations)

        # make the decision
        tree.make_decision(result_index)

        genome = tree.root.genome
        genome_eval = eval_func(genome)

        # if this eval is better than the best,
        # replace the best
        if genome_eval > best_eval:
            best_eval = genome_eval
            best_genome = genome
    
    return best_genome
