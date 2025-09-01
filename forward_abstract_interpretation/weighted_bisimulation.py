from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix


def is_lumpable(P, partition):
    """
    Check if the Markov chain with transition matrix P is lumpable with respect to the given partition.
    """
    # Create a mapping from state to block index
    state_to_block = {}
    for block_index, block in enumerate(partition):
        for state in block:
            state_to_block[state] = block_index

    # Check lumpability condition
    for block in partition:
        for i in block:
            for j in block:
                if i != j:
                    for target_block in partition:
                        sum_i = sum(P[i, k] for k in target_block)
                        sum_j = sum(P[j, k] for k in target_block)
                        if not np.isclose(sum_i, sum_j):
                            return False
    return True


def lumped_transition_matrix(P, partition):
    """
    Compute the lumped transition matrix for the Markov chain with transition matrix P and given partition.
    """
    m = len(partition)
    P_lumped = np.zeros((m, m))

    for i, block_i in enumerate(partition):
        for j, block_j in enumerate(partition):
            sum_prob = 0
            for state_i in block_i:
                for state_j in block_j:
                    sum_prob += P[state_i, state_j]
            P_lumped[i, j] = sum_prob / len(block_i)

    return P_lumped

def weighted_bisimulation(M):

    def saturation(s, A):
        return sum(M[s, k] for k in A)

    def is_splitter(C, X):
        for B in X:
            for x in B:
                for y in B:
                    if saturation(x, C) != saturation(y, C):
                        return True
        return False

    def split(C, X):
        partition = set()
        for B in X:
            sets = {}
            for x in B:
                sat = saturation(x, C)
                if sat not in sets:
                    sets[sat] = [x]
                else:
                    sets[sat].append(x)
            partition.update({frozenset(l) for l in sets.values()})
        return partition

    X = {frozenset(range(len(M)))}
    X1 = set()
    while True:
        changed = False
        X2 = X
        for C in X - X1:
            if is_splitter(C, X):
                X = split(C, X)
                X = {frozenset(l) for l in X}
                changed = True
        X1 = X2
        if not changed:
            X = sorted(list(s) for s in X)
            break
    return X


def lumpability(a: coo_matrix, direction: Literal['fw', 'bw', 'fwbw']):
    P = a.todense()
    if direction == 'fw':
        bisim = weighted_bisimulation(P)
        return bisim, lumped_transition_matrix(P, bisim)
    elif direction == 'bw':
        Q = P.transpose()
        bisim = weighted_bisimulation(Q)
        return bisim, lumped_transition_matrix(P, bisim)
    else:
        Q = P.transpose()
        bisim_fw = weighted_bisimulation(P)
        bisim_bw = weighted_bisimulation(Q)
        return (bisim_fw, lumped_transition_matrix(P, bisim_fw)), (bisim_bw, lumped_transition_matrix(P, bisim_bw))


if __name__ == '__main__':
    P = np.array([
        [1 / 3, 1 / 3, 1 / 3, 0, 0],
        [0, 1 / 3, 1 / 3, 0.40824828, 0],
        [0, 1 / 3, 1 / 3, 0.40824828, 0],
        [0, 0, 0, 1 / 2, 1 / 2],
        [0, 0.40824828, 0, 0, 1 / 2]
    ])
    # P = np.array([
    #     [1 / 2, 3 / 8, 1 / 16, 1 / 16],
    #     [7 / 16, 7 / 16, 0, 1 / 8],
    #     [1 / 16, 0, 1 / 2, 7 / 16],
    #     [0, 1 / 16, 3 / 8, 9 / 16]
    # ])

    print(weighted_bisimulation(P))
    print(lumped_transition_matrix(P, weighted_bisimulation(P)))