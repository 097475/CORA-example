import tensorflow as tf
from libmg import MGExplainer, SingleGraphLoader
from z3 import *

from old import interval_array_domain_intvals as domain
from old.interpreter import AbstractInterpreter
from wp_ops import tensor_to_interval, make_layer, relu, product, summation
from bwd_sym_interpreter import BWDSYMInterpreter
from wp import get_model, DatasetTest


def run_interpreter(expr, model, dataset, node, delta):
    graph = MGExplainer(model).explain(node, next(iter(SingleGraphLoader(dataset).load()))[0], None, False)
    n_nodes = graph.n_nodes
    n_classes = dataset.n_classes
    node_list = sorted(list(set(graph.a.row.tolist())))
    mapping = lambda xx: node_list.index(xx)
    print("Evaluating graph with ", n_nodes, " nodes.")

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

    abs_psi_dict = {'lin1': domain.make_layer(model.trainable_variables[0].value.numpy(), 'linear'),
                    'lin2': domain.make_layer(model.trainable_variables[1].value.numpy(), 'linear'),
                    'ReLU': domain.layer_relu}


    x = tf.convert_to_tensor(graph.x)
    x = domain.abstract(x, delta)

    print("Initial bound")
    print(x)

    e_tensor = tf.convert_to_tensor(graph.e)
    e = e_tensor
    # e = domain.abstract(e_tensor)
    y = tf.convert_to_tensor(graph.y)

    interp = AbstractInterpreter(domain, 'fw',
                                 abs_psi_dict,
                                 {'x': domain.prod}, {'+': domain.sm})

    print("Final labels")
    final_labels = interp.run(x, a, e, expr, save_history=True).x
    print(final_labels)

    # x = tf.convert_to_tensor(graph.x)
    Xarr = Array('X', IntSort(), ArraySort(IntSort(), RealSort()))
    post = Xarr[mapping(node)][0] > Xarr[mapping(node)][1]
    pre = tensor_to_interval(tf.convert_to_tensor(graph.x), delta, Xarr)
    #
    # a = tf.sparse.SparseTensor(
    #     tf.stack([tf.convert_to_tensor(list(map(mapping, graph.a.row)), dtype=tf.int64), tf.convert_to_tensor(list(map(mapping, graph.a.col)), dtype=tf.int64)], axis=1),
    #     tf.convert_to_tensor(graph.a.data),
    #     tf.convert_to_tensor((n_nodes, n_nodes), dtype=tf.int64)
    # )
    # e = tf.convert_to_tensor(graph.e)
    #
    interp = BWDSYMInterpreter({'lin1': make_layer(model.trainable_variables[0].value.numpy()),
                            'lin2': make_layer(model.trainable_variables[1].value.numpy()), 'ReLU': relu},
                           {'x': product}, {'+': summation})
    interp.run(Xarr, pre, post, n_classes, a, e, model.expr, final_labels, node)


def interpreter_debug():
    # Define dataset
    dataset = DatasetTest(True, True)

    # Define model
    channels = 5
    expr = '<x|+ ; lin1 ; ReLU ; <x|+ ; lin2'
    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    run_interpreter(expr, model, dataset, 1, 0.1)


if __name__ == '__main__':
    interpreter_debug()