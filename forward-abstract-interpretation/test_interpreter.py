import tensorflow as tf
import numpy as np
from libmg import Graph, Dataset, MGExplainer, Phi, Sigma, PsiLocal, CompilerConfig, MGCompiler, NodeConfig, EdgeConfig, SingleGraphLoader
from scipy.sparse import coo_matrix
from libmg import Graph, Dataset, print_labels
from arxiv.arxiv_mg import OGBDataset
from interpreter import AbstractInterpreter
import interval_array_domain_intvals as domain


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
    lin1 = PsiLocal.make('lin', tf.keras.layers.Dense(channels, activation='relu', use_bias=False))
    lin2 = PsiLocal.make('out', tf.keras.layers.Dense(classes, activation='softmax', use_bias=False))
    config = CompilerConfig.xae_config(NodeConfig(tf.float32, n_node_features), EdgeConfig(tf.float32, 1), tf.uint8, {})
    compiler = MGCompiler({'lin': lin1, 'out': lin2}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    model.build(n_node_features)
    return model


def run_interpreter(dataset, delta, channels, expr, model=None, node=None):
    n_classes = dataset.n_classes
    if model is not None and node is not None:
        original_y = dataset[0].y
        graph = MGExplainer(model).explain(node, next(iter(SingleGraphLoader(dataset).load()))[0], None, False)
        graph.y = original_y
        n_nodes = graph.n_nodes
        node_list = sorted(list(set(graph.a.row.tolist())))
        mapping = lambda xx: node_list.index(xx)
        rev_mapping = lambda xx: node_list[xx]

        a = tf.sparse.SparseTensor(
            tf.stack(
                [tf.convert_to_tensor(list(map(mapping, graph.a.row)), dtype=tf.int64), tf.convert_to_tensor(list(map(mapping, graph.a.col)), dtype=tf.int64)],
                axis=1),
            tf.convert_to_tensor(graph.a.data),
            tf.convert_to_tensor((n_nodes, n_nodes), dtype=tf.int64)
        )
        rev_a = tf.sparse.SparseTensor(
            tf.stack([tf.convert_to_tensor(graph.a.row, dtype=tf.int64), tf.convert_to_tensor(graph.a.col, dtype=tf.int64)], axis=1),
            tf.convert_to_tensor(graph.a.data),
            tf.convert_to_tensor(graph.a.shape, dtype=tf.int64)
        )

        abs_psi_dict = {'lin': domain.make_layer(model.trainable_variables[0].value.numpy(), 'RELU'),
                                  'out': domain.make_layer(model.trainable_variables[1].value.numpy(), 'softmax')}
    else:
        graph = dataset[0]
        a = tf.sparse.SparseTensor(
            tf.stack([tf.convert_to_tensor(graph.a.row, dtype=tf.int64), tf.convert_to_tensor(graph.a.col, dtype=tf.int64)], axis=1),
            tf.convert_to_tensor(graph.a.data),
            tf.convert_to_tensor(graph.a.shape, dtype=tf.int64)
        )
        rev_a, rev_mapping = None, None
        abs_psi_dict = {'lin': domain.make_layer(np.random.rand(dataset.n_node_features, channels), 'RELU'),
                                  'out': domain.make_layer(np.random.rand(channels, n_classes), 'softmax')}


    x = tf.convert_to_tensor(graph.x)
    x = domain.abstract(x, delta)

    e_tensor = tf.convert_to_tensor(graph.e)
    e = e_tensor
    # e = domain.abstract(e_tensor)
    y = tf.convert_to_tensor(graph.y)

    interp = AbstractInterpreter(domain, 'fw',
                                 abs_psi_dict,
                                 {'x': domain.prod}, {'+': domain.sm})

    print("Final labels")
    print(interp.run(x, a, e, expr, save_history=True).x)
    print("Middle labels")
    print(interp.get_labels(1))
    print(domain.concretize(interp.get_labels(1).x[0]))
    if model is not None and node is not None:
        for i in range(5):
            print_labels(domain.concretize(interp.get_labels(i).x[0]), rev_a, e_tensor, y, rev_mapping, 'test-' + str(i), open_browser=True, engine='pyvis')
    else:
        for i in range(5):
            print_labels(domain.concretize(interp.get_labels(i).x[0]), a, e_tensor, y, filename='test-' + str(i), open_browser=True, engine='pyvis')

def interpreter_debug():
    dataset = DatasetTest(True, True)

    expr = '<x|+ ; lin ; <x|+ ; out'
    channels = 5

    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)
    node = 2

    run_interpreter(dataset, 0.1, channels, expr, model, node)
    # run_interpreter(dataset, 0.1, channels, expr)


def interpreter_arxiv():
    dataset = OGBDataset("ogbn-arxiv", 'GCN')

    expr = '<x|+ ; lin ; <x|+ ; out'

    channels = 128

    model = get_model(expr, dataset.n_node_features, channels, 2)
    node = 2

    run_interpreter(dataset, 0.1, channels, expr, model, node)
    # run_interpreter(dataset, 0.1, channels, expr)


if __name__ == '__main__':
    interpreter_arxiv()