import math

import torch
from auto_LiRPA import BoundedModule
from libmg import Dataset, Phi, Sigma, PsiLocal, PsiGlobal, CompilerConfig, NodeConfig, EdgeConfig, MGCompiler, SingleGraphLoader, MultipleGraphLoader, MGExplainer, mg_reconstructor
from libmg.compiler.functions import FunctionDict
from ogb.nodeproppred import NodePropPredDataset
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.datasets import Citation, TUDataset, OGB
from spektral.transforms import OneHotLabels
from spektral.utils import gcn_filter
import numpy as np
import tensorflow as tf
import os
# os.environ['AUTOLIRPA_DEBUG'] = '1'
from keras import losses, optimizers, callbacks, layers, models, metrics
from tqdm import tqdm

from forward_abstract_interpretation import lirpa_domain
from lirpa_domain import LirpaInterpreter, interpreter, run_abstract_model
from run_interpreter import AbstractionSettings, NoAbstraction, EdgeAbstraction, BisimAbstraction


class DatasetTest(Dataset):
    g1 = (np.array([[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]], dtype=np.float32),
          coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1], ([0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 2, 3, 1, 3, 4, 1])),
                     shape=(5, 5), dtype=np.float32),
          np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.uint8))

    def __init__(self, edges=False, labels=False, **kwargs):
        self.edges = edges
        self.labels = labels
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        x1, a1, y1 = self.g1
        g1 = Graph(x1, a1, None, y1)
        graphs.append(g1)
        return graphs


class DatasetFigure(Dataset):
    g1 = (np.array([[0.5, 0.7], [-0.5, 1], [3.1, 2.3]], dtype=np.float32),
          coo_matrix(([1, 1, 1, 1], ([0, 0, 1, 2], [1, 2, 2, 1])),
                     shape=(3, 3), dtype=np.float32),
          np.array([[1, 0], [0, 1], [0, 1]], dtype=np.uint8))

    def __init__(self, edges=False, labels=False, **kwargs):
        self.edges = edges
        self.labels = labels
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        x1, a1, y1 = self.g1
        g1 = Graph(x1, a1, None, y1)
        graphs.append(g1)
        return graphs

# class DatasetFigure(Dataset):
#     g1 = (np.array([[0.5, 0.7], [-0.5, 1], [3.1, 2.3], [1, 1]], dtype=np.float32),
#           coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1, 1], ([0, 0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 3, 2, 3, 1, 3, 1, 2])),
#                      shape=(4, 4), dtype=np.float32),
#           np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.uint8))
#
#     def __init__(self, edges=False, labels=False, **kwargs):
#         self.edges = edges
#         self.labels = labels
#         super().__init__("libmg_test_dataset", **kwargs)
#
#     def read(self):
#         graphs = []
#         x1, a1, y1 = self.g1
#         g1 = Graph(x1, a1, None, y1)
#         graphs.append(g1)
#         return graphs

# class DatasetFigure(Dataset):
#     g1 = (np.array([[0.5, 0.7], [-0.5, 1]], dtype=np.float32),
#           coo_matrix(([1], ([0], [1])),
#                      shape=(2, 2), dtype=np.float32),
#           np.array([[1, 0], [0, 1]], dtype=np.uint8))
#
#     def __init__(self, edges=False, labels=False, **kwargs):
#         self.edges = edges
#         self.labels = labels
#         super().__init__("libmg_test_dataset", **kwargs)
#
#     def read(self):
#         graphs = []
#         x1, a1, y1 = self.g1
#         g1 = Graph(x1, a1, None, y1)
#         graphs.append(g1)
#         return graphs


class OGBDataset(OGB):
    def __init__(self, dataset, **kwargs):
        # Setting name
        self.name = dataset.name
        # Setting masks
        idx = dataset.get_idx_split()
        idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
        n_nodes = dataset.graph['num_nodes']
        self.mask_tr = np.zeros(n_nodes, dtype=bool)
        self.mask_va = np.zeros(n_nodes, dtype=bool)
        self.mask_te = np.zeros(n_nodes, dtype=bool)
        self.mask_tr[idx_tr] = True
        self.mask_va[idx_va] = True
        self.mask_te[idx_te] = True
        # Setting labels for one hot representation
        dataset.labels = np.squeeze(dataset.labels)
        # Super call
        transforms = kwargs.pop('transforms', [])
        super().__init__(dataset, transforms=[OneHotLabels(depth=dataset.num_classes)] + transforms, **kwargs)

def preprocess_gcn_mg(graph):
    new_a = gcn_filter(graph.a)
    graph.e = np.expand_dims(new_a.data, axis=-1)
    new_a.data = np.ones_like(new_a.data)
    graph.a = new_a.tocoo()
    return graph

class CastTo:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, graph):
        if graph.x is not None and graph.x.dtype != self.dtype:
            graph.x = graph.x.astype(self.dtype)
        if graph.a is not None and graph.a.dtype != self.dtype:
            graph.a = graph.a.astype(self.dtype)
        if graph.e is not None and graph.e.dtype != self.dtype:
            graph.e = graph.e.astype(self.dtype)
        if graph.y is not None and graph.y.dtype != self.dtype:
            graph.y = graph.y.astype(self.dtype)
        return graph


def print_dataset_info(dataset):
    print("Dataset name:", dataset.name)
    print("Number of graphs:", dataset.n_graphs)
    print("Number and type of node features:", dataset.n_node_features, dataset[0].x.dtype)
    print("Number and type of edge features:", dataset.n_edge_features, dataset[0].e.dtype if dataset.n_edge_features is not None else None)
    print("Number and type of labels:", dataset.n_labels, dataset[0].y.dtype)
    print("Type of adjacency matrix:", dataset[0].a.dtype)
    if dataset.n_graphs == 1:
        print("For single-graph datasets")
        print("Number of nodes:", dataset[0].n_nodes)
        print("Number of edges:", dataset[0].n_edges)


def train_node_level(dataset, model, epochs, patience, learning_rate):

    def mask_to_weights(mask):
        return mask.astype(np.float32) / np.count_nonzero(mask)

    weights_tr, weights_va, weights_te = (
        mask_to_weights(mask)
        for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=losses.CategoricalCrossentropy(reduction="sum", from_logits=True),
        weighted_metrics=["acc"],
    )

    # Train model
    loader_tr = SingleGraphLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleGraphLoader(dataset, sample_weights=weights_va)
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
    )

    # Evaluate model
    print("Evaluating model.")
    loader_te = SingleGraphLoader(dataset, sample_weights=weights_te)
    eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


def train_graph_level(dataset, model, epochs, patience, learning_rate):
    idxs = np.random.permutation(len(dataset))
    split_tr, split_va = int(0.8 * len(dataset)), int(0.1 * len(dataset))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_tr, split_tr + split_va])
    dataset_tr, dataset_va, dataset_te = dataset[idx_tr], dataset[idx_va], dataset[idx_te]
    #dataset_tr, dataset_va, dataset_te = dataset[:int(0.5*dataset.n_graphs)], dataset[int(0.5*dataset.n_graphs): int(0.8*dataset.n_graphs)], dataset[int(0.8*dataset.n_graphs):dataset.n_graphs]
    loader_tr, loader_va, loader_te = MultipleGraphLoader(dataset_tr, node_level=False, shuffle=False), MultipleGraphLoader(dataset_va, node_level=False, shuffle=False), MultipleGraphLoader(dataset_te, node_level=False, shuffle=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        weighted_metrics=["acc"],
    )

    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
    )

    # Evaluate model
    print("Evaluating model.")
    eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


def get_gcn(dataset, expr):
    # Define mg model
    prod = Phi(lambda i, e, j: i * e)
    sm = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))
    dense = PsiLocal.make_parametrized('dense', lambda channels: tf.keras.layers.Dense(int(channels), activation='relu', use_bias=True))
    out = PsiLocal.make('out', tf.keras.layers.Dense(dataset.n_labels, activation='linear', use_bias=True))
    sum_pool = PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='SumPooling')
    mean_pool = PsiGlobal(single_op=lambda x: tf.reduce_mean(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='MeanPooling')
    if len(dataset) > 1:
        config = CompilerConfig.xaei_config(NodeConfig(tf.float32, dataset.n_node_features), EdgeConfig(tf.float32, 1), tf.float32, {'float': 0.000001})
    else:
        config = CompilerConfig.xae_config(NodeConfig(tf.float32, dataset.n_node_features), EdgeConfig(tf.float32, 1), tf.float32, {'float': 0.000001})
    compiler = MGCompiler({'dense': dense, 'out': out, 'sum': sum_pool, 'mean': mean_pool}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    # model.build(dataset.n_node_features)
    return model


def train_or_load_weights(dataset, model, epochs, patience, learning_rate, retrain):
    directory = dataset.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + '/' + 'model.weights.h5'

    if os.path.exists(path) and not retrain:
        print("Loading weights.")
        model.load_weights(path)
        print("Evaluating model.")
        loader = MultipleGraphLoader(dataset, node_level=False, shuffle=False) if len(dataset) > 1 else SingleGraphLoader(dataset)
        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss=losses.CategoricalCrossentropy(from_logits=True),
            weighted_metrics=["acc"],
        )
        eval_results = model.evaluate(loader.load(), steps=loader.steps_per_epoch)
        print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
    else:
        if len(dataset) == 1:
            train_node_level(dataset, model, epochs, patience, learning_rate)
        else:
            train_graph_level(dataset, model, epochs, patience, learning_rate)

        print("Saving weights.")
        model.save_weights(path, overwrite=True)

def get_abstract_model(model, abs_settings):
    interpreter.set_concrete_layers(model.mg_layers)
    interpreter.set_graph_abstraction(abs_settings.graph_abstraction)
    # if abs_settings.graph_abstraction.optimize_memory:
    #     interpreter.hops = 2
    return interpreter.run(model.expr)

def print_bounds(lb, ub, pred, truth):
    pred = pred[0]
    pred_classes = np.argmax(pred.numpy(), axis=1)
    truth_classes = np.argmax(truth.numpy(), axis=1)
    # lb = lb[0] if lb.shape[0] == 1 else lb
    # ub = ub[0] if ub.shape[0] == 1 else ub
    n_nodes = lb.shape[0]
    n_classes = lb.shape[1]  # eq. to n_node_features
    for i in range(n_nodes):
        print(f'Node {i} top-1 prediction {pred_classes[i]} ground-truth {truth_classes[i]}')
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == truth_classes[i] else ''
            print('f_{j}(x_0): {l:8.3f} <= {p:8.3f} <= {u:8.3f} {ind}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), p=pred[i][j], ind=indicator))
    print()


def check_soundness(pred, lb, ub):
    eps = interpreter.atol
    pred = pred[0] if isinstance(pred, tuple) else pred
    pred = pred[0] if pred.shape.ndims == 3 else pred
    lb = lb[0] if lb.ndim == 3 else lb
    ub = ub[0] if ub.ndim == 3 else ub
    for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
        for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
            assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
    print('Soundness check passed')


def is_overlapping(intv1, intv2):
    return intv1[0] <= intv2[1] and intv1[1] >= intv2[0]


def run_graph_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True):
    loader = MultipleGraphLoader(dataset, node_level=False, shuffle=False, epochs=1, batch_size=1)

    for graph_tf, graph_np in zip(loader.load(), dataset):
        # Conc exe
        (conc_x, conc_a, conc_e, conc_i), y = graph_tf
        out = model((conc_x, conc_a, conc_e, conc_i))
        prediction = (out[0][0].numpy())

        ### Setting up graph
        abs_x, abs_a, abs_e = abs_settings.abstract(graph_np)

        ### Run
        abs_lb, abs_ub = run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm)

        ### Concretize
        lb, ub = abs_settings.concretize(abs_lb, abs_ub)

        abs_prediction_lb = lb[0]
        abs_prediction_ub = ub[0]
        true_label = graph_np.y

        if not skip_uncorrect or np.argmax(prediction) == np.argmax(true_label):
            idx = np.argmax(true_label)
            true_label_lb, true_label_ub = abs_prediction_lb[idx], abs_prediction_ub[idx]
            overlaps = False
            for i in range(len(true_label)):
                if i != idx and is_overlapping((true_label_lb, true_label_ub), (abs_prediction_lb[i], abs_prediction_ub[i])):
                    print("Failed to prove the property")
                    overlaps = True
                    break
            if not overlaps:
                print("Property proved")
        else:
            print("Skipping because the model failed to predict correctly")


def run_node_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True):
    graph_tf = SingleGraphLoader(dataset, epochs=1).load().__iter__().__next__()
    graph_np = dataset[0]
    true_labels = graph_np.y

    # Conc exe
    (conc_x, conc_a, conc_e), y = graph_tf
    out = model((conc_x, conc_a, conc_e))
    predictions = out[0].numpy()

    for node in tqdm(range(graph_np.n_nodes)):
        # Generate cut graph
        new_graph = MGExplainer(model).explain(node, (conc_x, conc_a, conc_e), None, False)
        print(new_graph.n_nodes)
        node_list = sorted(list(set(new_graph.a.row.tolist())))
        mapping = lambda xx: node_list.index(xx)
        mapped_a = coo_matrix((new_graph.a.data,(np.array(list(map(mapping, new_graph.a.row))), np.array(list(map(mapping, new_graph.a.col))))), shape=(new_graph.n_nodes, new_graph.n_nodes))
        new_graph.a = mapped_a

        ### Setting up graph
        abs_x, abs_a, abs_e = abs_settings.abstract(new_graph)

        # import timeit
        # print(timeit.repeat(lambda: run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm), repeat=3, number=1))

        ### Run

        abs_lb, abs_ub = run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm)


        ### Concretize
        lb, ub = abs_settings.concretize(abs_lb, abs_ub)

        abs_prediction_lb = lb
        abs_prediction_ub = ub

        print(abs_prediction_lb, abs_prediction_ub)

        if not skip_uncorrect or np.argmax(predictions[node]) == np.argmax(true_labels[node]):
            idx = np.argmax(true_labels[node])
            abs_prediction_lb_node, abs_prediction_ub_node = abs_prediction_lb[mapping(node)], abs_prediction_ub[mapping(node)]
            true_label_lb, true_label_ub = abs_prediction_lb_node[idx], abs_prediction_ub_node[idx]
            overlaps = False
            for i in range(len(true_labels[node])):
                if i != idx and is_overlapping((true_label_lb, true_label_ub), (abs_prediction_lb_node[i], abs_prediction_ub_node[i])):
                    print("Failed to prove the property")
                    overlaps = True
                    break
            if not overlaps:
                print("Property proved")
        else:
            print("Skipping because the model failed to predict correctly")


# def run_node_task_gcn_optimized(model, abstract_model, dataset, abs_settings, skip_uncorrect=True):
#     graph_tf = SingleGraphLoader(dataset, epochs=1).load().__iter__().__next__()
#     graph_np = dataset[0]
#     true_labels = graph_np.y
#
#
#     prod = Phi(lambda i, e, j: i * e)
#     sm = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))
#     dense = PsiLocal.make_parametrized('dense', lambda channels: tf.keras.layers.Dense(int(channels), activation='relu', use_bias=True))
#     out = PsiLocal.make('out', tf.keras.layers.Dense(dataset.n_labels, activation='linear', use_bias=True))
#     sum_pool = PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='SumPooling')
#     mean_pool = PsiGlobal(single_op=lambda x: tf.reduce_mean(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='MeanPooling')
#
#     weights = model.weights
#     n_gcn_layers = len(weights) // 2
#     exprs = [mg_reconstructor.reconstruct(model.expr)] + [';'.join(['<x|+ ; dense[32]'] * i) for i in range(1, n_gcn_layers)]
#     abs_exprs = ['<x|+ ; dense[32]' for _ in range(n_gcn_layers-1)] + ['<x|+ ; out']
#     cut_models = [get_gcn(dataset, expr) for expr in exprs]
#     sizes = [dataset.n_node_features, 32]
#     abs_models = []
#     i = 0
#     for expr, input_size in zip(abs_exprs, sizes):
#         config = CompilerConfig.xae_config(NodeConfig(tf.float32, input_size), EdgeConfig(tf.float32, 1), tf.float32, {'float': 0.000001})
#         compiler = MGCompiler({'dense': dense, 'out': out, 'sum': sum_pool, 'mean': mean_pool}, {'+': sm}, {'x': prod}, config)
#         expr_model = compiler.compile(expr)
#         model_weights = weights[i:i+2]
#         expr_model.set_weights(model_weights)
#         abstract_model_t = get_abstract_model(expr_model, abs_settings)
#         abs_models.append(abstract_model_t)
#         i += 2
#
#     # Conc exe
#     (conc_x, conc_a, conc_e), y = graph_tf
#     out = model((conc_x, conc_a, conc_e))
#     predictions = out[0].numpy()
#
#     for node in tqdm(range(0, graph_np.n_nodes)):
#         mapping = None
#         for i, (cut_model, abs_model) in enumerate(zip(cut_models, abs_models)):
#             # Generate cut graph
#             new_graph = MGExplainer(cut_model).explain(node, (conc_x, conc_a, conc_e), None, False)
#             print(new_graph.n_nodes)
#             old_mapping = mapping
#             node_list = sorted(list(set(new_graph.a.row.tolist())))
#             mapping = lambda xx: node_list.index(xx)
#             mapped_a = coo_matrix((new_graph.a.data,(np.array(list(map(mapping, new_graph.a.row))), np.array(list(map(mapping, new_graph.a.col))))), shape=(new_graph.n_nodes, new_graph.n_nodes))
#             new_graph.a = mapped_a
#
#             ### Setting up graph
#             abs_x, abs_a, abs_e = abs_settings.abstract(new_graph)
#             if i > 0:
#                 x_lb = lb[[old_mapping(n) for n in node_list]]
#                 x_ub = ub[[old_mapping(n) for n in node_list]]
#                 abs_x = lirpa_domain.abstract_x(np.zeros_like(x_lb), x_L=x_lb.numpy(), x_U=x_ub.numpy())
#
#
#             # import timeit
#             # print(timeit.repeat(lambda: run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm), repeat=3, number=1))
#
#             ### Run
#             lirpa_model = BoundedModule(abs_model, (torch.empty_like(abs_x), abs_a, torch.empty_like(abs_e)), device=abs_a.device, verbose=True)
#             lb, ub, A = lirpa_model.compute_bounds(x=(abs_x, abs_a, abs_e), method=abs_settings.algorithm, IBP=True, return_A=True, need_A_only=True, needed_A_dict={lirpa_model.final_name: ['/x.1', '/e.1']})
#             abs_lb, abs_ub = lb.detach()[0], ub.detach()[0]
#
#
#             ### Concretize
#             lb, ub = abs_settings.concretize(abs_lb, abs_ub)
#
#         abs_prediction_lb = lb
#         abs_prediction_ub = ub
#
#         print(abs_prediction_lb, abs_prediction_ub)
#
#         if not skip_uncorrect or np.argmax(predictions[node]) == np.argmax(true_labels[node]):
#             idx = np.argmax(true_labels[node])
#             abs_prediction_lb_node, abs_prediction_ub_node = abs_prediction_lb[mapping(node)], abs_prediction_ub[mapping(node)]
#             true_label_lb, true_label_ub = abs_prediction_lb_node[idx], abs_prediction_ub_node[idx]
#             overlaps = False
#             for i in range(len(true_labels[node])):
#                 if i != idx and is_overlapping((true_label_lb, true_label_ub), (abs_prediction_lb_node[i], abs_prediction_ub_node[i])):
#                     print("Failed to prove the property")
#                     overlaps = True
#                     break
#             if not overlaps:
#                 print("Property proved")
#         else:
#             print("Skipping because the model failed to predict correctly")
#         break

# Node task
def figure_setup():
    dataset = DatasetFigure(transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[2] ; <x|+; out')
    model.set_weights([np.array([[1., 1.], [-1., 1]]), np.array([0., 0.]), np.array([[1., 1.], [-1., 1]]), np.array([0., 0.])])
    model.summary()

    abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(optimized_gcn=True), 'backward')
    # abs_settings = AbstractionSettings(0.1, 0.,  EdgeAbstraction({(0, 0), (1, 1), (2, 2)}, True, edge_label_generator='GCN', optimized_gcn=True), 'IBP')
    # abs_settings = AbstractionSettings(0, 0, BisimAbstraction('bw', optimized_gcn=True), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)
    # run_node_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=False)


    for concrete_graph_np, concrete_graph_tf in zip(dataset, MultipleGraphLoader(dataset, node_level=True, epochs=1, shuffle=False).load()):
        #####
        # concrete_graph_np.e = np.array([[1/3], [1/(math.sqrt(3) * math.sqrt(2)) * 2 / 3], [1/(math.sqrt(3) * math.sqrt(2)) * 2 / 3], [1/(math.sqrt(3) * math.sqrt(2)) * 2 / 3], [1/3], [1/3 ], [1/3 ], [1/3 ], [1/3 ], [1/3 ], [1/3 ], [1/3 ], [1/3 ]], dtype=np.float32)

        ### Setting up graph
        abs_x, abs_a, abs_e = abs_settings.abstract(concrete_graph_np)

        ### Run
        abs_lb, abs_ub = run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm)

        ### Concretize
        lb, ub = abs_settings.concretize(abs_lb, abs_ub)

        (conc_x, conc_a, conc_e, _), y = concrete_graph_tf
        # conc_e = tf.constant(concrete_graph_np.e)

        pred = model((conc_x, conc_a, conc_e))
        print(pred)
        print(lb, ub)
        check_soundness(pred, lb, ub)
        print_bounds(lb, ub, pred, y)


# Node task
def debug_setup():
    dataset = DatasetTest(transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; out')
    model.summary()
    model.load_weights('debug.weights.h5')

    # abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(optimized_gcn=True), 'backward')
    abs_settings = AbstractionSettings(0, 0, BisimAbstraction('fw'), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)
    # run_node_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=False)

    for concrete_graph_np, concrete_graph_tf in zip(dataset, MultipleGraphLoader(dataset, node_level=True, epochs=1, shuffle=False).load()):
        ### Setting up graph
        abs_x, abs_a, abs_e = abs_settings.abstract(concrete_graph_np)

        ### Run
        abs_lb, abs_ub = run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm)

        ### Concretize
        lb, ub = abs_settings.concretize(abs_lb, abs_ub)

        (conc_x, conc_a, conc_e, _), y = concrete_graph_tf

        pred = model((conc_x, conc_a, conc_e))
        print(pred)
        print(lb, ub)
        check_soundness(pred, lb, ub)
        print_bounds(lb, ub, pred, y)

# Node task
def arxiv_setup():
    dataset = OGBDataset(NodePropPredDataset('ogbn-arxiv'), transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; out')
    model.summary()

    train_or_load_weights(dataset, model, epochs=200, patience=10, learning_rate=1e-2, retrain=False)

    abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(optimized_gcn=True), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)

    run_node_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True)

# Node task
def cora_setup():
    dataset = Citation('cora', transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; out')
    model.summary()

    train_or_load_weights(dataset, model, epochs=200, patience=10, learning_rate=1e-2, retrain=False)

    abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(optimized_gcn=True), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)

    run_node_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True)
    # run_node_task_gcn_optimized(model, abstract_model, dataset, abs_settings, skip_uncorrect=True)

# Graph task
def enzymes_setup():
    dataset = TUDataset('ENZYMES', transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; dense[32] ; <x|+ ; dense[32] ;  sum ; dense[128] ; out')
    model.summary()

    train_or_load_weights(dataset, model, epochs=200, patience=10, learning_rate=1e-2, retrain=False)

    abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)

    run_graph_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True)

# Graph task
def proteins_setup():
    dataset = TUDataset('PROTEINS_full', transforms=[preprocess_gcn_mg, CastTo(np.float32)])
    # TODO: remove last three features
    print_dataset_info(dataset)

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; dense[32] ; <x|+ ; dense[32] ;  mean ; dense[128] ; out')
    model.summary()

    train_or_load_weights(dataset, model, epochs=200, patience=10, learning_rate=1e-2, retrain=False)

    abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)

    run_graph_task(model, abstract_model, dataset, abs_settings, skip_uncorrect=True)


if __name__ == '__main__':
    figure_setup()
    # for _ in range(1):
    # debug_setup()
    # arxiv_setup()
    # cora_setup()
    # enzymes_setup()
    # proteins_setup()