import numpy as np

def one_point_crossover(genes1: np.ndarray, genes2: np.ndarray, point: int):
    '''
        Combine genes; cutting at one point and swapping.
    '''
    return np.concatenate([genes1[:point], genes2[point:]])


def two_point_crossover(genes1: np.ndarray, genes2: np.ndarray, point1: int, point2: InterruptedError):
    '''
        Combine genes; cutting at two points and swapping.
    '''
    lower = min(point1, point2)
    upper = max(point1, point2)
    return np.concatenate([genes1[:lower], genes2[lower:upper], genes1[upper:]])
