import numpy as np
import math
import random


def ucb1(total_value, sims, total_sims_of_parent, constant = 2.0):
    if sims == 0:
        return float('inf')
    return (total_value / sims) + (constant * math.sqrt(math.log(total_sims_of_parent) / sims))


def get_next_moves(genome):
    '''
        Return a list of all possible moves from that position.
        In the case of a genome, it is a list of all possible next genome states.
    '''

    # not changing at all counts as a decision
    next_moves = [np.copy(genome)]
    for i in range(len(genome)):

        # pick an index to switch, flip the bit, add to list
        new_genome = np.copy(genome)
        new_genome[i] = not new_genome[i]
        next_moves.append(new_genome)
    return next_moves


class Node:

    def __init__(self, parent, genome) -> None:
        '''
            Initialize all variables.
        '''
        self.parent = parent
        self.total_value = 0
        self.num_sims = 0
        self.children = None
        self.genome = genome

    def expand(self):
        '''
            Create the child nodes.
        '''
        next_moves = get_next_moves(self.genome)
        self.children = [Node(self, new_genome) for new_genome in next_moves]
        random.shuffle(self.children)

    def pass_result_back(self, value):
        '''
            Pass rollout results back.
        '''
        self.total_value += value
        self.num_sims += 1
        if self.parent is not None:
            self.parent.pass_result_back(value)
    
    def get_highest_ucb1_child(self):
        '''
            Get the child with the highest UCB1 value.
        '''
        highest_index = 0
        highest_ucb1 = float("-inf")
        for i, child in enumerate(self.children):
            ucb1_value = ucb1(child.total_value, child.num_sims, self.num_sims)
            if ucb1_value > highest_ucb1:
                highest_ucb1 = ucb1_value
                highest_index = i
        return self.children[highest_index]

    def rollout(self, eval_func):
        '''
            Normally there is a rollout here,
            which involves a series of random decisions,
            followed by an evaluation.
            It works better with this particular problem
            if you just return an evaluation at this node.
        '''
        return eval_func(self.genome)
    
    def child_with_most_sims(self):
        '''
            Find the child that the most simulations have been run on.
            Returns the index of that child.
        '''

        highest_num = 0
        highest_index = 0
        for i, child in enumerate(self.children):
            num_sims = child.num_sims
            if num_sims > highest_num:
                highest_num = num_sims
                highest_index = i
        return highest_index
