import abc
import math
from copy import deepcopy
from typing import Literal

import numpy as np
from functools import reduce

import torch
from libmg import MGExplainer, SingleGraphLoader
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.utils import normalized_adjacency, degree_power
from tqdm import tqdm
import tensorflow as tf

from forward_abstract_interpretation import lirpa_domain
from forward_abstract_interpretation.spanning_tree import generate_uncertain_edge_set
from weighted_bisimulation import lumpability


def edge_abstraction(graph: Graph, uncertain_edges: set[tuple[int, int]], add_missing_edges: bool, edge_label_generator: tuple[float, float] | Literal['GCN'] = (0, 1)):
    new_x = graph.x # x stays the same
    new_y = graph.y  # y stays the same

    # Write 1 in the matrix for certain edges, -1 for uncertain ones
    if not add_missing_edges:
        new_a = deepcopy(graph.a)
        for i, e in enumerate(zip(*graph.a.coords)):
            if e in uncertain_edges:
                new_a.data[i] = -1.
    else:
        new_a = coo_matrix(([-1]*math.prod(graph.a.shape),
                            ([i for i in range(graph.a.shape[0]) for _ in range(graph.a.shape[1])],
                             [i % graph.a.shape[0] for i in range(math.prod(graph.a.shape))])),
                   shape=graph.a.shape, dtype=graph.a.dtype)
        for i, e in enumerate(zip(*new_a.coords)):
            if e not in uncertain_edges:
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
        # assert {(i, i) for i in range(graph.n_nodes)}.issubset(certain_edges) # self loops must always be certain
        a_lb_data = np.where(new_a.data == -1, 1, new_a.data)
        a_lb = deepcopy(new_a)
        a_lb.data = a_lb_data
        new_e_lb = np.expand_dims(normalized_adjacency(a_lb).data, 1)

        a_ub_data = np.where(new_a.data == -1, 0, new_a.data)
        a_ub = deepcopy(new_a)
        a_ub.data = a_ub_data
        # upper bound for uncertain edges can be computed by 1/sqrt(1+crt_i) * 1/sqrt(crt_j)
        # upper bound for certain edges can be computed by 1/sqrt(crt_i) * 1/sqrt(crt_j)
        crt = degree_power(a_ub, 1).data.squeeze(axis=0)
        new_e_ub = np.zeros_like(new_e_lb)
        for k, e in enumerate(zip(*new_a.coords)):
            i, j = e
            if e in uncertain_edges:
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
    a.data = graph.e.squeeze(axis=-1)
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


class AbstractionSettings:
    def __init__(self, node_delta, edge_delta, graph_abstraction, algorithm):
        self.node_delta = node_delta
        self.edge_delta = edge_delta
        self.graph_abstraction = graph_abstraction
        self.algorithm = algorithm

    def abstract(self, graph):
        return self.graph_abstraction.abstract(graph, node_delta=self.node_delta, edge_delta=self.edge_delta)

    def abstract_node(self, graph, model, node, graph_tf):
        return self.graph_abstraction.abstract_node(graph, node_delta=self.node_delta, edge_delta=self.edge_delta, model=model, node=node, graph_tf=graph_tf)

    def concretize(self, abs_lb, abs_ub):
        return self.graph_abstraction.concretize(abs_lb, abs_ub)


class GraphAbstraction(abc.ABC):
    @abc.abstractmethod
    def abstract(self, graph, node_delta, edge_delta):
        raise NotImplementedError

    @abc.abstractmethod
    def concretize(self, abs_lb, abs_ub):
        raise NotImplementedError

    @abc.abstractmethod
    def handle_pooling(self, x):
        raise NotImplementedError


class NoAbstraction(GraphAbstraction):
    def __init__(self, optimized_gcn=False):
        self.optimized_gcn = optimized_gcn

    def abstract(self, graph, node_delta, edge_delta):
        abs_graph, y = (lirpa_domain.abstract_x(graph.x, node_delta), lirpa_domain.abstract_adj(graph.a), lirpa_domain.abstract_e(graph.e, edge_delta)), graph.y
        ####################
        if self.optimized_gcn:
            new_a_low = deepcopy(graph.a)
            new_a_high = deepcopy(graph.a)
            new_a_low.data = np.squeeze(graph.e, axis=-1) - edge_delta
            new_a_high.data = np.squeeze(graph.e, axis=-1) + edge_delta
            e_low = np.expand_dims(new_a_low.todense(), axis=0)
            e_high = np.expand_dims(new_a_high.todense(), axis=0)
            e = np.zeros_like(e_low)
            abs_e = lirpa_domain.abstract_e(e, x_L=e_low, x_U=e_high)
            a = abs_graph[1][0]
            edge_status = torch.zeros_like(abs_e.data)
            edge_status[0][a[0], a[1]] = a[2].float()
            abs_graph = abs_graph[0], edge_status, abs_e
        ###################
        return abs_graph

    def concretize(self, abs_lb, abs_ub):
        return abs_lb, abs_ub

    def handle_pooling(self, x):
        return x



class EdgeAbstraction(GraphAbstraction):

    def __init__(self, uncertain_edges: set[tuple[int, int]] | float, add_missing_edges: bool, edge_label_generator: tuple[int, int] | str, optimized_gcn=False):
        super().__init__()
        self.uncertain_edges = uncertain_edges if isinstance(uncertain_edges, set) else set()
        self.ratio = uncertain_edges if isinstance(uncertain_edges, float) else None
        self.add_missing_edges = add_missing_edges
        self.edge_label_generator = edge_label_generator
        self.optimized_gcn = optimized_gcn


    def abstract_node(self, graph, node_delta, edge_delta, model, node, graph_tf):
        conc_x, conc_a, conc_e = graph_tf
        # Generate cut graph
        new_graph = MGExplainer(model).explain(node, (conc_x, conc_a, conc_e), None, False)
        tqdm.write(f"\nAnalyzing graph with {new_graph.n_nodes} nodes")
        conc_graph_n_nodes = new_graph.n_nodes

        node_list = sorted(list(set(new_graph.a.row.tolist())))
        mapping = lambda xx: node_list.index(xx)
        mapped_a = coo_matrix((new_graph.a.data,(np.array(list(map(mapping, new_graph.a.row))), np.array(list(map(mapping, new_graph.a.col))))), shape=(new_graph.n_nodes, new_graph.n_nodes))
        new_graph.a = mapped_a

        # Regenerate uncertain edge set after cut
        self.generate_uncertain_edge_set(new_graph.a)
        self.uncertain_edges = {(node_list[i], node_list[j]) for (i, j) in self.uncertain_edges}

        (x, a, (e_lb, e_ub)), y = edge_abstraction(graph, self.uncertain_edges, self.add_missing_edges, self.edge_label_generator)

        cut_low = MGExplainer(model).explain(node, (conc_x, conc_a, tf.convert_to_tensor(e_lb)), None, False)
        cut_high = MGExplainer(model).explain(node, (conc_x, conc_a, tf.convert_to_tensor(e_ub)), None, False)

        x = new_graph.x
        a  = new_graph.a
        e_lb = cut_low.e
        e_ub = cut_high.e

        abs_x = lirpa_domain.abstract_x(x, node_delta)
        abs_a = lirpa_domain.abstract_adj(a)
        abs_e = lirpa_domain.abstract_e(np.zeros_like(e_lb), delta=edge_delta, x_L=e_lb, x_U=e_ub)
        abs_graph = abs_x, abs_a, abs_e
        ####################
        if self.optimized_gcn:
            new_a_low = deepcopy(a)
            new_a_high = deepcopy(a)
            new_a_low.data = np.squeeze(e_lb,axis=-1) - edge_delta
            new_a_high.data = np.squeeze(e_ub, axis=-1) + edge_delta
            unrolled_edge_status = abs_a[0][2].numpy()

            # edge abstraction
            new_a_low.data[unrolled_edge_status < 0] = np.clip(new_a_low.data[unrolled_edge_status < 0], a_max=0, a_min=None)
            new_a_high.data[unrolled_edge_status < 0] = np.clip(new_a_high.data[unrolled_edge_status < 0], a_min=0, a_max=None)

            e_low = np.expand_dims(new_a_low.todense(), axis=0)
            e_high = np.expand_dims(new_a_high.todense(), axis=0)
            e = np.zeros_like(e_low)
            abs_e = lirpa_domain.abstract_e(e, x_L=e_low, x_U=e_high)

            edge_status = torch.zeros(a.shape).unsqueeze(0)
            edge_status[0][abs_a[0][0], abs_a[0][1]] = abs_a[0][2].float()

            abs_graph = abs_graph[0], edge_status, abs_e
        ###################
        return abs_graph, mapping, conc_graph_n_nodes


    def abstract(self, graph, node_delta, edge_delta):
        (x, a, (e_lb, e_ub)), y = edge_abstraction(graph, self.uncertain_edges, self.add_missing_edges, self.edge_label_generator)
        abs_x = lirpa_domain.abstract_x(x, node_delta)
        abs_a = lirpa_domain.abstract_adj(a)
        abs_e = lirpa_domain.abstract_e(np.zeros_like(e_lb), delta=edge_delta, x_L=e_lb, x_U=e_ub)
        abs_graph = abs_x, abs_a, abs_e
        ####################
        if self.optimized_gcn:
            new_a_low = deepcopy(a)
            new_a_high = deepcopy(a)
            new_a_low.data = np.squeeze(e_lb,axis=-1) - edge_delta
            new_a_high.data = np.squeeze(e_ub, axis=-1) + edge_delta
            unrolled_edge_status = abs_a[0][2].numpy()

            # edge abstraction
            new_a_low.data[unrolled_edge_status < 0] = np.clip(new_a_low.data[unrolled_edge_status < 0], a_max=0, a_min=None)
            new_a_high.data[unrolled_edge_status < 0] = np.clip(new_a_high.data[unrolled_edge_status < 0], a_min=0, a_max=None)

            e_low = np.expand_dims(new_a_low.todense(), axis=0)
            e_high = np.expand_dims(new_a_high.todense(), axis=0)
            e = np.zeros_like(e_low)
            abs_e = lirpa_domain.abstract_e(e, x_L=e_low, x_U=e_high)

            edge_status = torch.zeros(a.shape).unsqueeze(0)
            edge_status[0][abs_a[0][0], abs_a[0][1]] = abs_a[0][2].float()

            abs_graph = abs_graph[0], edge_status, abs_e
        ###################
        return abs_graph

    def concretize(self, abs_lb, abs_ub):
        return abs_lb, abs_ub

    def handle_pooling(self, x):
        return x

    def generate_uncertain_edge_set(self, a):
        self.uncertain_edges = generate_uncertain_edge_set(self.ratio, a)


class BisimAbstraction(GraphAbstraction):
    def __init__(self, direction, optimized_gcn=False):
        super().__init__()
        self.direction = direction
        self.bisim_map = None
        self.optimized_gcn = optimized_gcn

    def abstract_node(self, graph, node_delta, edge_delta, model, node, graph_tf):
        conc_x, conc_a, conc_e = graph_tf
        # Generate cut graph
        new_graph = MGExplainer(model).explain(node, (conc_x, conc_a, conc_e), None, False)
        tqdm.write(f"\nAnalyzing graph with {new_graph.n_nodes} nodes")
        conc_graph_n_nodes = new_graph.n_nodes

        node_list = sorted(list(set(new_graph.a.row.tolist())))
        mapping = lambda xx: node_list.index(xx)
        mapped_a = coo_matrix((new_graph.a.data, (np.array(list(map(mapping, new_graph.a.row))), np.array(list(map(mapping, new_graph.a.col))))), shape=(new_graph.n_nodes, new_graph.n_nodes))
        new_graph.a = mapped_a

        bisim_map, ((abs_x_lb, abs_x_ub), abs_a, abs_e), y = bisim_abstraction(new_graph, self.direction)
        self.bisim_map = bisim_map
        abs_graph = lirpa_domain.abstract_x(np.zeros_like(abs_x_lb), delta=node_delta, x_L=abs_x_lb, x_U=abs_x_ub), lirpa_domain.abstract_adj(abs_a), lirpa_domain.abstract_e(abs_e, edge_delta)
        ####################
        if self.optimized_gcn:
            new_a_low = deepcopy(abs_a)
            new_a_high = deepcopy(abs_a)
            new_a_low.data = np.squeeze(abs_e, axis=-1) - edge_delta
            new_a_high.data = np.squeeze(abs_e, axis=-1) + edge_delta
            e_low = np.expand_dims(new_a_low.todense(), axis=0)
            e_high = np.expand_dims(new_a_high.todense(), axis=0)
            e = np.zeros_like(e_low)
            abs_e = lirpa_domain.abstract_e(e, x_L=e_low, x_U=e_high)
            a = abs_graph[1][0]
            edge_status = torch.zeros_like(abs_e.data)
            edge_status[0][a[0], a[1]] = a[2].float()
            abs_graph = abs_graph[0], edge_status, abs_e
        ###################
        return abs_graph, mapping, conc_graph_n_nodes

    def abstract(self, graph, node_delta, edge_delta):
        bisim_map, ((abs_x_lb, abs_x_ub), abs_a, abs_e), y = bisim_abstraction(graph, self.direction)
        self.bisim_map = bisim_map
        abs_graph = lirpa_domain.abstract_x(np.zeros_like(abs_x_lb), delta=node_delta, x_L=abs_x_lb, x_U=abs_x_ub), lirpa_domain.abstract_adj(abs_a), lirpa_domain.abstract_e(abs_e, edge_delta)
        ####################
        if self.optimized_gcn:
            new_a_low = deepcopy(abs_a)
            new_a_high = deepcopy(abs_a)
            new_a_low.data = np.squeeze(abs_e, axis=-1) - edge_delta
            new_a_high.data = np.squeeze(abs_e, axis=-1) + edge_delta
            e_low = np.expand_dims(new_a_low.todense(), axis=0)
            e_high = np.expand_dims(new_a_high.todense(), axis=0)
            e = np.zeros_like(e_low)
            abs_e = lirpa_domain.abstract_e(e, x_L=e_low, x_U=e_high)
            a = abs_graph[1][0]
            edge_status = torch.zeros_like(abs_e.data)
            edge_status[0][a[0], a[1]] = a[2].float()
            abs_graph = abs_graph[0], edge_status, abs_e
        ###################
        return abs_graph

    def concretize(self, abs_lb, abs_ub):
        indexes = sorted(self.bisim_map.orig_to_part_mapping.values())
        if abs_lb.shape[1] > 1:
            lb = abs_lb[indexes]
            ub = abs_ub[indexes]
        else:  # Pooling output
            lb = abs_lb
            ub = abs_ub
        return lb, ub

    def handle_pooling(self, x):
        return self.bisim_map.multiply(x)
