import math
from copy import deepcopy
from typing import Literal

import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.utils import normalized_adjacency, degree_power

from weighted_bisimulation import lumpability


def edge_abstraction(graph: Graph, certain_edges: set[tuple[int, int]], add_missing_edges: bool, edge_label_generator: tuple[float, float] | Literal['GCN'] = (0, 1)):
    new_x = graph.x # x stays the same
    new_y = graph.y  # y stays the same

    # Write 1 in the matrix for certain edges, -1 for uncertain ones
    if not add_missing_edges:
        new_a = deepcopy(graph.a)
        for i, e in enumerate(zip(*graph.a.coords)):
            if e not in certain_edges:
                new_a.data[i] = -1.
    else:
        new_a = coo_matrix(([-1]*math.prod(graph.a.shape),
                            ([i for i in range(graph.a.shape[0]) for _ in range(graph.a.shape[1])],
                             [i % graph.a.shape[0] for i in range(math.prod(graph.a.shape))])),
                   shape=graph.a.shape, dtype=graph.a.dtype)
        for i, e in enumerate(zip(*new_a.coords)):
            if e in certain_edges:
                new_a.data[i] = 1.

    # Generate new edge labels
    if isinstance(edge_label_generator, tuple): # Keep labels for existing edges, set new edges to tuple bounds
        edge_label_dict = {e: graph.e[i] for i, e in enumerate(zip(*graph.a.coords))}
        default_lb, default_ub = edge_label_generator
        new_e_lb = np.full((len(new_a.data), graph.e.shape[1]), default_lb, dtype=np.float32)
        new_e_ub = np.full((len(new_a.data), graph.e.shape[1]), default_ub, dtype=np.float32)
        for i, e in enumerate(zip(*new_a.coords)):
            if e in edge_label_dict:
                new_e_lb[i] = edge_label_dict[e]
                new_e_ub[i] = edge_label_dict[e]
    elif edge_label_generator == 'GCN':     # For a safe bound, consider upper bound = 1.0 and lower bound = 1/n_nodes
        assert {(i, i) for i in range(graph.n_nodes)}.issubset(certain_edges) # self loops must always be certain
        a_lb_data = np.where(new_a.data == -1, 1, new_a.data)
        a_lb = deepcopy(new_a)
        a_lb.data = a_lb_data
        new_e_lb = np.expand_dims(normalized_adjacency(a_lb).data, 1)

        a_ub_data = np.where(new_a.data == -1, 0, new_a.data)
        a_ub = deepcopy(new_a)
        a_ub.data = a_ub_data
        # upper bound for uncertain edges can be computed by 1/sqrt(1+crt_i) * 1/sqrt(crt_j)
        # upper bound for certain edges can be computed by 1/sqrt(crt_i) * 1/sqrt(crt_j)
        crt = degree_power(a_ub, 1).data.squeeze()
        new_e_ub = np.zeros_like(new_e_lb)
        for k, e in enumerate(zip(*new_a.coords)):
            i, j = e
            if e not in certain_edges:
                new_e_ub[k] = np.array([1/math.sqrt(1 + crt[i]) * 1/math.sqrt(crt[j])])
            else:
                new_e_ub[k] = np.array([1 / math.sqrt(crt[i]) * 1 / math.sqrt(crt[j])])
    else:
        raise NotImplementedError

    return (new_x, new_a, (new_e_lb, new_e_ub)), new_y


class BisimulationMap:
    def __init__(self, bisimulation):
        self.orig_to_part_mapping = {v: i for i, partition in enumerate(bisimulation) for v in partition}
        self.multiplicity = list(self.orig_to_part_mapping.values())

    def multiply(self, x):
        return x[:, self.multiplicity]

def regenerate_graph(graph, bisimulation, lumped_matrix):
    x, a, e = graph.x, graph.a, graph.e
    n_node_features = x.shape[1]
    # n_edge_features = e.shape[1]

    new_n_nodes = len(bisimulation)
    new_x_lb = np.zeros((new_n_nodes, n_node_features), dtype=np.float32)
    new_x_ub = np.zeros((new_n_nodes, n_node_features), dtype=np.float32)
    for i, partition in enumerate(bisimulation):
        node_feats = [x[node] for node in partition]
        for j in range(n_node_features):
            values = [l[j] for l in node_feats]
            glb = reduce(min, values)
            lub = reduce(max, values)
            new_x_lb[i][j] = glb
            new_x_ub[i][j] = lub

    # e_dict = {}
    # for k, (i, j) in enumerate(zip(*a.coords)):
    #     label = e[k]
    #     new_i, new_j = orig_to_part_mapping[i], orig_to_part_mapping[j]
    #     if (new_i, new_j) not in e_dict:
    #         e_dict[(new_i, new_j)] = [label]
    #     else:
    #         e_dict[(new_i, new_j)].append(label)
    lumped_edge_features = lumped_matrix[lumped_matrix != 0]
    edges = sorted([tuple(coord) for coord in np.argwhere(lumped_matrix)])
    # edges = list(set((orig_to_part_mapping[i], orig_to_part_mapping[j]) for (i, j) in zip(*a.coords)))

    sources, targets = zip(*edges)
    new_n_edges = len(edges)
    assert new_n_edges == len(lumped_edge_features)
    new_a = coo_matrix(([1] * new_n_edges, (sources, targets)), shape=(new_n_nodes, new_n_nodes), dtype=np.float32)

    # new_e_lb = np.zeros((new_n_edges, n_edge_features))
    # new_e_ub = np.zeros((new_n_edges, n_edge_features))
    new_e = np.expand_dims(np.array(lumped_edge_features, dtype=np.float32), -1)
    # for i, edge in enumerate(edges):
    #     edge_feats = lumped_edge_features[i]
    #     for j in range(n_edge_features):
    #         values = [l[j] for l in edge_feats]
    #         glb = reduce(min, values)
    #         lub = reduce(max, values)
    #         new_e_lb[i][j] = glb
    #         new_e_ub[i][j] = lub
    return (new_x_lb, new_x_ub), new_a, new_e


def bisim_abstraction(graph: Graph, direction: Literal['fw', 'bw', 'fwbw']):
    a = deepcopy(graph.a)
    a.data = graph.e.squeeze()
    if direction == 'fw':
        bisim, lumped_matrix = lumpability(a, direction)
    elif direction == 'bw':
        bisim, lumped_matrix = lumpability(a, direction)
    else:
        (bisim_fw, lumped_matrix_fw), (bisim_bw, lumped_matrix_bw) = lumpability(a, direction)
        if bisim_fw != bisim_bw:
            raise Exception('bisim_fw and bisim_bw are not equal', bisim_fw, bisim_bw)
        else:
            bisim = bisim_fw
            lumped_matrix = lumped_matrix_fw
    return BisimulationMap(bisim), regenerate_graph(graph, bisim, lumped_matrix), graph.y


# def bisim_abstraction(graph: Graph, epsilon, mode: Literal['nodes', 'edges', 'both']):
#     digits = -(math.log10(epsilon) + 1)
#     x, a, e = graph.x, graph.a, graph.e
#     n_nodes = x.shape[0]
#     n_node_features = x.shape[1]
#     n_edges = e.shape[0]
#     n_edge_features = e.shape[1]
#     if mode == 'nodes':
#         nx_graph = nx.DiGraph(a)
#         # nodes with the same label are put in the same initial partition
#         uniques, inverse = np.unique(x.round(decimals=int(digits)), return_inverse=True, axis=0)
#         num_partitions = len(uniques)
#         partitions = [() for _ in range(num_partitions)] # partitions are lists of tuples
#         for i in range(n_nodes):
#             partitions[inverse[i]] += (i, )
#         bisimulation = bispy.dovier_piazza_policriti(nx_graph, initial_partition=partitions, is_integer_graph=True)
#         return regenerate_graph(x, a, e, bisimulation)
#
#     else:
#         temp_features = max(n_node_features, n_edge_features) + 1 # 1/0 to mark nodes edges, features
#         temp_x = np.zeros((n_nodes + n_edges, temp_features))
#         temp_x[:n_nodes] = np.hstack((np.ones((n_nodes, 1)), x,
#                                       np.zeros((n_nodes, temp_features - (n_node_features + 1)))))
#         temp_edges = []
#         for k, (i, j) in enumerate(zip(*a.coords)):
#             label = np.hstack((np.array([0], dtype=np.float32), e[k])) # 0 to mark edge
#             temp_x[n_nodes+k] = np.hstack((label, np.zeros((temp_features - (n_edge_features + 1)))))
#             temp_edges.append((i, n_nodes + k))
#             temp_edges.append((n_nodes + k, j))
#         temp_sources, temp_targets = zip(*sorted(temp_edges))
#         temp_a = coo_matrix(([1] * len(temp_edges), (temp_sources, temp_targets)),
#                             shape=(temp_x.shape[0], temp_x.shape[0]), dtype=np.float32)
#         nx_graph = nx.DiGraph(temp_a)
#         if mode == 'edges':
#             uniques, inverse = np.unique(temp_x[n_nodes:].round(decimals=int(digits)), return_inverse=True, axis=0)
#             num_partitions = len(uniques)
#             partitions =  [() for _ in range(num_partitions)] + [tuple(i for i in range(n_nodes))]
#             for i in range(n_edges):
#                 partitions[inverse[i]] += (n_nodes + i,)
#         else:
#             uniques, inverse = np.unique(temp_x.round(decimals=int(digits)), return_inverse=True, axis=0)
#             num_partitions = len(uniques)
#             partitions = [() for _ in range(num_partitions)]
#             for i in range(n_edges+n_nodes):
#                 partitions[inverse[i]] += (i,)
#         bisimulation = bispy.dovier_piazza_policriti(nx_graph, initial_partition=partitions, is_integer_graph=True)
#         bisimulation = [partition for partition in bisimulation if all(i < n_nodes for i in partition)]
#         return regenerate_graph(x, a, e, bisimulation)
#
#                                                     # n_nodes x n_node_features             # n_edges x n_edge_features
# def merge_abstraction(graphs: list[Graph]) -> tuple[(np.ndarray, np.ndarray), coo_matrix, (np.ndarray,np.ndarray)]:
#     xs, As, es = [], [], []
#     for g in graphs:
#         xs.append(g.x)
#         As.append(g.a)
#         es.append(g.e)
#
#     # merged node features
#     n_nodes = reduce(lambda x, y: max(x, y.shape[0]), xs, 0)
#     n_node_features = reduce(lambda x, y: max(x, y.shape[1]), xs, 0)
#     new_x_lb = np.zeros((n_nodes, n_node_features))
#     new_x_ub = np.zeros((n_nodes, n_node_features))
#     for i in range(n_nodes):
#         for j in range(n_node_features):
#             values = [x[i][j] for x in xs if x.shape[0] > i and x.shape[1] > j]
#             glb = reduce(min, values)
#             lub = reduce(max, values)
#             new_x_lb[i][j] = glb
#             new_x_ub[i][j] = lub
#
#     # merged edge features
#     disjoint_edge_sets = [set(zip(*a.coords)) for a in As]
#     merged_edge_dict = {k: [] for k in set.union(*disjoint_edge_sets)}
#     merged_edge_list = sorted(merged_edge_dict.keys())
#     certain_edges = set.intersection(*disjoint_edge_sets)
#     n_edges = len(merged_edge_dict)
#     n_edge_features = reduce(lambda x, y: max(x, y.shape[1]), es, 0)
#
#     for a, e in zip(As, es):
#         for i, edge in enumerate(zip(*a.coords)):
#             merged_edge_dict[edge].append(e[i])
#
#     new_e1 = np.ones((n_edges, 1))
#
#     new_e2_lb = np.zeros((n_edges, n_edge_features))
#     new_e_lb = np.hstack((new_e1, new_e2_lb))
#
#     new_e2_ub = np.zeros((n_edges, n_edge_features))
#     new_e_ub = np.hstack((new_e1, new_e2_ub))
#
#     for i, edge in enumerate(merged_edge_list):
#         if edge not in certain_edges:
#             new_e_lb[i][0] = 0. # edge is uncertain in both lb and ub arrays
#             new_e_ub[i][0] = 0.
#         edge_feats = merged_edge_dict[edge]
#         for j in range(n_edge_features):
#             values = [e[j] for e in edge_feats if e.shape[0] > j]
#             glb = reduce(min, values)
#             lub = reduce(max, values)
#             new_e_lb[i][j+1] = glb
#             new_e_ub[i][j+1] = lub
#
#
#     # merged adj matrix
#     sources, targets = zip(*merged_edge_list)
#     new_a = coo_matrix(([1] * n_edges, (sources, targets)), shape=(n_nodes, n_nodes), dtype=np.float32)
#
#     return (new_x_lb, new_x_ub), new_a, (new_e_lb, new_e_ub)
