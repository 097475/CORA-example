import random

import numpy as np

def minimum_spanning_tree(adj_matrix, start=0):
    n = adj_matrix.shape[0]
    visited = [False] * n
    visited[start] = True
    edges = set()

    # Use a queue for BFS (could also use a stack for DFS)
    queue = [start]

    while queue:
        u = queue.pop(0)  # pop from front for BFS
        for v in range(n):
            if (adj_matrix[u, v] == 1 or adj_matrix[v, u] == 1) and not visited[v]:
                visited[v] = True
                edges.add((min(u, v), max(u, v)))
                queue.append(v)

    return edges

def generate_uncertain_edge_set(ratio, adj):
    max_n_edges = len(adj.data)
    edges = set(zip(*adj.coords))
    adj = adj.todense()
    n_nodes = adj.shape[0]
    indegrees = adj.sum(axis=0)
    outdegrees = adj.sum(axis=1)  # Sum rows
    start_node = np.argmax(indegrees + outdegrees)
    mst = minimum_spanning_tree(adj, start_node)
    forbidden_edges = mst.union({(i, i) for i in range(n_nodes)})
    assert forbidden_edges.issubset(edges)
    uncertain_edges = set()
    uncertain_edges_to_generate = int(max_n_edges * ratio)
    pickable_edges = list(edges.difference(forbidden_edges))
    while len(uncertain_edges) < uncertain_edges_to_generate and len(pickable_edges) > 0:
        idx = random.randrange(len(pickable_edges))
        e = pickable_edges.pop(idx)
        uncertain_edges.add(e)
    return uncertain_edges





