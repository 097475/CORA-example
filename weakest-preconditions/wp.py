import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
from libmg import Graph, Dataset, MGExplainer, Phi, Sigma, PsiLocal, CompilerConfig, MGCompiler, NodeConfig, EdgeConfig, SingleGraphLoader
from arxiv.arxiv_mg import OGBDataset
from z3 import *

from wp_ops import tensor_to_interval, make_layer, idn, relu, product, summation
from wp_interpreter import WPInterpreter


class DatasetTest(Dataset):
    g1 = (np.array([[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]], dtype=np.float32),
          coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], [0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 1, 4])),
                     shape=(5, 5), dtype=np.float32),
          np.array([[0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.40824828], [0.3333333], [0.3333333], [0.40824828], [0.49999997],
                    [0.49999997], [0.40824828], [0.49999997]], np.float32),
          np.array([[2], [4], [8], [2], [2]], dtype=np.uint8))

    def __init__(self, edges=False, labels=False, **kwargs):
        self.edges = edges
        self.labels = labels
        self.n_classes = 2
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        x1, a1, e1, y1 = self.g1
        g1 = Graph(x1, a1, e1 if self.edges else None, y1 if self.labels else None)
        graphs.append(g1)
        return graphs


def get_model(expr, n_node_features, channels, classes):
    prod = Phi(lambda i, e, j: i * e)
    sm = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))
    lin1 = PsiLocal.make('lin1', tf.keras.layers.Dense(channels, use_bias=False))
    lin2 = PsiLocal.make('lin2', tf.keras.layers.Dense(classes, use_bias=False))
    sfmax = PsiLocal.make('sfmax', tf.keras.activations.softmax)
    ReLU = PsiLocal.make('ReLU', tf.keras.activations.relu)
    config = CompilerConfig.xae_config(NodeConfig(tf.float32, n_node_features), EdgeConfig(tf.float32, 1), tf.uint8, {})
    compiler = MGCompiler({'lin1': lin1, 'lin2': lin2, 'sfmax': sfmax, 'ReLU': ReLU}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    model.build(n_node_features)
    return model


def run_interpreter(model, dataset, node, delta):
    graph = MGExplainer(model).explain(node, next(iter(SingleGraphLoader(dataset).load()))[0], None, False)
    n_nodes = graph.n_nodes
    n_classes = dataset.n_classes
    node_list = sorted(list(set(graph.a.row.tolist())))
    mapping = lambda xx: node_list.index(xx)
    print("Evaluating graph with ", n_nodes, " nodes.")

    x = tf.convert_to_tensor(graph.x)
    Xarr = Array('X', IntSort(), ArraySort(IntSort(), RealSort()))
    post = Xarr[mapping(node)][0] > Xarr[mapping(node)][1]
    pre = tensor_to_interval(x, delta, Xarr)

    a = tf.sparse.SparseTensor(
        tf.stack([tf.convert_to_tensor(list(map(mapping, graph.a.row)), dtype=tf.int64), tf.convert_to_tensor(list(map(mapping, graph.a.col)), dtype=tf.int64)], axis=1),
        tf.convert_to_tensor(graph.a.data),
        tf.convert_to_tensor((n_nodes, n_nodes), dtype=tf.int64)
    )
    e = tf.convert_to_tensor(graph.e)

    interp = WPInterpreter({'lin1': make_layer(model.trainable_variables[0].value.numpy()),
                            'lin2': make_layer(model.trainable_variables[1].value.numpy()), 'sfmax': idn, 'ReLU': relu},
                           {'x': product}, {'+': summation})
    interp.run(Xarr, pre, post, n_classes, a, e, model.expr)


def interpreter_debug():
    # Define dataset
    dataset = DatasetTest(True, True)

    # Define model
    channels = 5
    expr = '<x|+ ; lin1 ; ReLU ; <x|+ ; lin2 ; sfmax'
    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    run_interpreter(model, dataset, 2, 0.1)


def interpreter_arxiv():
    # Define dataset
    dataset = OGBDataset("ogbn-arxiv", 'GCN')

    # Define model
    channels = 128  # Number of channels for GCN layers
    expr = '<x|+ ; lin1 ; sfmax ; <x|+ ; lin2 ; sfmax'
    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    run_interpreter(model, dataset, 2, 0.1)


if __name__ == '__main__':
    interpreter_arxiv()
    # interpreter_debug()