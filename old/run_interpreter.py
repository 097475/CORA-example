import abc
import os
from copy import deepcopy

import tensorflow as tf
import numpy as np
import torch
from libmg import Phi, Sigma, PsiLocal, CompilerConfig, MGCompiler, NodeConfig, EdgeConfig, MultipleGraphLoader, PsiGlobal, MGExplainer
from ogb.nodeproppred import Evaluator
from scipy.sparse import coo_matrix
from libmg import Graph, Dataset
from spektral.datasets import Citation, MNIST
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess

# from forward_abstract_interpretation.arxiv.arxiv_mg import OGBDataset
import lirpa_domain
from arxiv.arxiv_mg import OGBDataset2
from graph_abstraction import edge_abstraction, bisim_abstraction
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class DatasetTest(Dataset):
    g1 = (np.array([[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]], dtype=np.float32),
          coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], [0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 1, 4])),
                     shape=(5, 5), dtype=np.float32),
          np.array([[0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.40824828], [0.3333333], [0.3333333], [0.40824828], [0.49999997],
                    [0.49999997], [0.40824828], [0.49999997]], np.float32),
          np.array([[0], [1], [1], [0], [0]], dtype=np.uint8))

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


def dataset_to_gcn_and_sparse_y(dataset):
    if dataset.a is not None:
        dataset.a = dataset.a.tocoo()  # somehow the matrix is in csr format, scipy 1.13.1 is needed
        for i in range(len(dataset)):
            edge_features = tf.expand_dims(tf.cast(dataset.a.data, tf.float32), axis=1).numpy()
            dataset[i].e = edge_features
            dataset[i].x = dataset[i].x.astype(np.float32)
            if not np.isscalar(dataset[i].y) and len(dataset[i].y) > 1:
                dataset[i].y = np.expand_dims(np.argmax(dataset[i].y, axis=1), axis=1)
        dataset.a.data = np.ones_like(dataset.a.data)
    else:
        for i in range(len(dataset)):
            dataset[i].a = dataset[i].a.tocoo()  # somehow the matrix is in csr format, scipy 1.13.1 is needed
            graph = dataset[i]
            edge_features = tf.expand_dims(tf.cast(graph.a.data, tf.float32), axis=1).numpy()
            dataset[i].e = edge_features
            dataset[i].a.data = np.ones_like(graph.a.data)
            dataset[i].x = dataset[i].x.astype(np.float32)
            if not np.isscalar(dataset[i].y) and len(dataset[i].y) > 1:
                dataset[i].y = np.expand_dims(np.argmax(dataset[i].y, axis=1), axis=1)
    return dataset


def check_soundness(pred, lb, ub):
    eps = 0.000001
    pred = pred[0] if isinstance(pred, tuple) else pred
    pred = pred[0] if pred.shape.ndims == 3 else pred
    lb = lb[0] if lb.ndim == 3 else lb
    ub = ub[0] if ub.ndim == 3 else ub
    for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
        for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
            assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
    print('Soundness check passed')


def print_bounds(lb, ub, pred, truth):
    pred = pred[0]
    pred_classes = np.argmax(pred.numpy(), axis=1)
    truth_classes = tf.squeeze(truth, axis=1).numpy()
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


# def concrete_prediction(pred, node):
#     return tf.argmax(pred[0][node]).numpy().item()
#
#
# def abstract_prediction(lb, ub, node):
#     node_lb = lb[0][node]
#     node_ub = ub[0][node]
#     if node_lb[0] <= node_ub[1] and node_lb[1] <= node_ub[0]:
#         return None
#     if node_lb[0] >= node_ub[1]:
#         return 0
#     if node_lb[1] >= node_ub[0]:
#         return 1


def get_model(expr, n_node_features, channels, classes, weights=None):
    prod = Phi(lambda i, e, j: i * e)
    sm = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))
    lin1 = PsiLocal.make('lin', tf.keras.layers.Dense(channels, activation='relu', use_bias=True))
    lin2 = PsiLocal.make('out', tf.keras.layers.Dense(classes, activation='linear', use_bias=True))
    sum_pool = PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), name='SumPooling')
    mean_pool = PsiGlobal(single_op=lambda x: tf.reduce_mean(x, axis=0, keepdims=False), name='MeanPooling')
    config = CompilerConfig.xae_config(NodeConfig(tf.float32, n_node_features), EdgeConfig(tf.float32, 1), tf.uint8, {'float': 0.000001})
    compiler = MGCompiler({'lin': lin1, 'out': lin2, 'sum': sum_pool, 'mean': mean_pool}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    model.build(n_node_features)
    if weights is not None:
        model.set_weights(weights)
    return model


def train_model(model, dataset, lr, epochs):
    optimizer = Adam(learning_rate=lr)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    # Training function
    @tf.function
    def train(inputs, target, mask):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)[0]
            loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions

    # Evaluation with OGB
    evaluator = Evaluator("ogbn-arxiv")

    def evaluate(preds, y, masks, evaluator):
        p = preds.numpy().argmax(-1)[:, None]
        tr_mask, va_mask, te_mask = masks
        tr_auc = evaluator.eval({"y_true": y[tr_mask], "y_pred": p[tr_mask]})["acc"]
        va_auc = evaluator.eval({"y_true": y[va_mask], "y_pred": p[va_mask]})["acc"]
        te_auc = evaluator.eval({"y_true": y[te_mask], "y_pred": p[te_mask]})["acc"]
        return tr_auc, va_auc, te_auc

    graph = dataset[0]
    x = tf.convert_to_tensor(graph.x)
    adj = tf.sparse.SparseTensor(
        tf.stack([tf.convert_to_tensor(graph.a.row, dtype=tf.int64), tf.convert_to_tensor(graph.a.col, dtype=tf.int64)], axis=1),
        tf.convert_to_tensor(graph.a.data),
        tf.convert_to_tensor(graph.a.shape, dtype=tf.int64)
    )
    e = tf.convert_to_tensor(graph.e)
    y = graph.y
    mask_tr = dataset.tr_mask
    masks = [dataset.tr_mask, dataset.va_mask, dataset.te_mask]

    if not os.path.isfile('./my_model.weights.h5'):
        for i in range(1, 1 + epochs):
            tr_loss, predictions = train([x, adj, e], y, mask_tr)
            tr_acc, va_acc, te_acc = evaluate(predictions, y, masks, evaluator)
            print(
                "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val acc: {:.3f} - Test acc: "
                "{:.3f}".format(i, tr_loss, tr_acc, va_acc, te_acc)
            )

        # Evaluate model
        print("Evaluating model.")
        tr_acc, va_acc, te_acc = evaluate(model([x, adj, e], training=False)[0], y, masks, evaluator)
        print("Done! - Test acc: {:.3f}".format(te_acc))

        print("Saving weights.")
        model.save_weights("my_model.weights.h5")

    print("Loading weights.")
    model.load_weights("my_model.weights.h5")

    print("Running model")
    preds = model([x, adj, e], training=False)[0]

    print('Evaluating again.')
    _, _, te_acc = evaluate(preds, y, masks, evaluator)
    print("Done! - Test acc: {:.3f}".format(te_acc))

    # print(tf.argmax(preds, axis=1).numpy())
    # print(np.squeeze(y))



def run_interpreter(dataset, delta, domain, model, graph_abstraction='node_labels', node=None):
    loader = MultipleGraphLoader(dataset, epochs=1, shuffle=False)
    for graph in loader.load():
        x = graph[0][0]
        a = graph[0][1]
        e = graph[0][2]
        y = graph[1] if len(graph[1]) == len(x) else tf.expand_dims(tf.repeat(graph[1], len(x)), axis=1)
        if node is not None:
            original_y = y
            new_graph = MGExplainer(model).explain(node, (x, a, e), None, False)
            new_graph.y = original_y
            n_nodes = new_graph.n_nodes
            node_list = sorted(list(set(new_graph.a.row.tolist())))
            mapping = lambda xx: node_list.index(xx)
            rev_mapping = lambda xx: node_list[xx]

            # a_tensor = tf.sparse.SparseTensor(
            #     tf.stack(
            #         [tf.convert_to_tensor(list(map(mapping, new_graph.a.row)), dtype=tf.int64), tf.convert_to_tensor(list(map(mapping, new_graph.a.col)), dtype=tf.int64)],
            #         axis=1),
            #     tf.convert_to_tensor(new_graph.a.data),
            #     tf.convert_to_tensor((n_nodes, n_nodes), dtype=tf.int64)
            # )
            a_tensor = coo_matrix((new_graph.a.data,(np.array(list(map(mapping, new_graph.a.row))), np.array(list(map(mapping, new_graph.a.col))))), shape=(n_nodes, n_nodes))
            # rev_a = tf.sparse.SparseTensor(
            #     tf.stack([tf.convert_to_tensor(new_graph.a.row, dtype=tf.int64), tf.convert_to_tensor(new_graph.a.col, dtype=tf.int64)], axis=1),
            #     tf.convert_to_tensor(new_graph.a.data),
            #     tf.convert_to_tensor(new_graph.a.shape, dtype=tf.int64)
            # )
            # x = tf.convert_to_tensor(new_graph.x, dtype=tf.float32)
            x = graph.x
            a = domain.abstract_adj(a_tensor)
            # e = tf.convert_to_tensor(new_graph.e, dtype=tf.float32)
            e = graph.e
            # rev_a = domain.abstract_graph(rev_a)
            y = tf.gather(y, node_list).numpy()

            abs_psi_dict = {'lin': domain.make_layer(model.trainable_variables[0].value.numpy(), 'RELU'),
                            'out': domain.make_layer(model.trainable_variables[1].value.numpy(), 'linear')}
        else:
            # graph = dataset[0]
            # a_tensor = tf.sparse.SparseTensor(
            #     tf.stack([tf.convert_to_tensor(graph.a.row, dtype=tf.int64), tf.convert_to_tensor(graph.a.col, dtype=tf.int64)], axis=1),
            #     tf.convert_to_tensor(graph.a.data),
            #     tf.convert_to_tensor(graph.a.shape, dtype=tf.int64)
            # )
            a_tensor = a
            a = domain.abstract_adj(a_tensor)
            # y = tf.convert_to_tensor(graph.y)
            rev_a, rev_mapping = None, None
            abs_psi_dict = {'lin': domain.make_layer(model.trainable_variables[0].value.numpy(), 'RELU') if len(model.trainable_variables) > 0 else None,
                            'out': domain.make_layer(model.trainable_variables[1].value.numpy(), 'linear') if len(model.trainable_variables) > 1 else None,
                            'g0': torch.nn.Linear(3, 1)}

        if graph_abstraction == 'node_labels':
            x_tensor = x
            x = domain.abstract_x(x_tensor, delta)

            e_tensor = e
            e = domain.abstract_e(e_tensor)
        elif graph_abstraction == 'node+edge_labels':
            x_tensor = x
            x = domain.abstract_x(x_tensor, delta)

            e_tensor = e
            e = domain.abstract_e(e_tensor, delta)


        interp = lirpa_domain.interpreter
        interp.set_psi(abs_psi_dict)

        lb, ub = domain.run_abstract_model(interp.run(model.expr), x, a, e, 'backward')
        pred = model((x_tensor, a_tensor, e_tensor))
        print(pred)
        print(lb, ub)
        check_soundness(pred, lb, ub)
        print_bounds(lb, ub, pred, y)



def new_run_interpreter(dataset, model, abs_settings, node=None):
    ### Setting up interpreter
    # abs_psi_dict = {'lin': lirpa_domain.make_layer(model.trainable_variables[0].value.numpy(), model.trainable_variables[1].value.numpy(), 'RELU') if len(model.trainable_variables) > 0 else None,
    #                 'out': lirpa_domain.make_layer(model.trainable_variables[2].value.numpy(), model.trainable_variables[3].value.numpy(), 'softmax') if len(model.trainable_variables) > 2 else None,
    #                 'sum': lirpa_domain.make_pooling('sum'),
    #                 'mean': lirpa_domain.make_pooling('mean')}

    interpreter = lirpa_domain.interpreter
    interpreter.set_concrete_layers(model.mg_layers)
    interpreter.set_graph_abstraction(abs_settings.graph_abstraction)

    ### Setting up graph
    concrete_graph = dataset[0]
    abs_x, abs_a, abs_e = abs_settings.abstract(concrete_graph)

    ### Run
    abs_lb, abs_ub = lirpa_domain.run_abstract_model(interpreter.run(model.expr), abs_x, abs_a, abs_e, abs_settings.algorithm)

    ### Concretize
    lb, ub = abs_settings.concretize(abs_lb, abs_ub)


    (conc_x, conc_a, conc_e, _), y = MultipleGraphLoader(dataset, epochs=1, shuffle=False).load().__iter__().__next__()
    pred = model((conc_x, conc_a, conc_e))
    print(pred)
    print(lb, ub)
    check_soundness(pred, lb, ub)
    print_bounds(lb, ub, pred, y)
    



def interpreter_arxiv():
    dataset = OGBDataset2("ogbn-arxiv", 'GCN')

    expr = '<x|+ ; lin ; <x|+ ; out'

    channels = 128

    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    train_model(model, dataset, 0.001, 500)

    node = 0

    import lirpa_domain as domain_lirpa
    run_interpreter(dataset, 0.1, domain_lirpa, model, node=node)


def interpreter_cora():
    dataset = Citation('cora', normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
    dataset.n_classes = dataset.n_labels
    dataset = dataset_to_gcn_and_sparse_y(dataset)

    expr = '<x|+ ; <x|+'

    channels = 5

    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    # train_model(model, dataset, 0.001, 500)

    node = None

    import lirpa_domain as domain
    run_interpreter(dataset, 0.1, domain, model, node)


def interpreter_MNIST():
    dataset = MNIST(p_flip=0.0, k=8, transforms=[LayerPreprocess(GCNConv)])[:5]
    dataset = dataset_to_gcn_and_sparse_y(dataset)
    dataset.n_classes = 10
    for g in dataset:
        g.a = dataset.a

    expr = '<x|+ ; lin ; <x|+ ; out'
    channels = 5
    weights = [np.ones((dataset.n_node_features, channels), dtype=np.float32), np.ones((channels, dataset.n_classes), dtype=np.float32)]
    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes, weights)
    node = 2

    import lirpa_domain as domain_lirpa
    run_interpreter(dataset, 0.1, domain_lirpa, model, node)


def safe_ptb():
    dataset = DatasetTest(True, True)

    expr = '<x|+ ; lin ; <x|+ ; out'
    channels = 5

    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes)

    import lirpa_domain as domain_lirpa

    find_safe_ptb(model, dataset, domain_lirpa, 2)

def interpreter_debug():
    dataset = DatasetTest(True, True)

    expr = '<x|+ ; lin ; <x|+ ; out'
    # expr = '(g0 || <x|+);(<x|+ | |x>+);<x|+'
    # expr = '|x>+ ; lin ; |x>+'
    # expr = '|x>+'
    # expr = 'sum'
    channels = 5
    # weights = [np.ones((dataset.n_node_features, channels), dtype=np.float32), np.ones((channels, dataset.n_classes), dtype=np.float32)]
    # weights = [np.array([[ 0.0738678 , -0.34869242,  0.11053455,  0.1412316 , -0.83844304], [-0.4333606 ,  0.6358581 , -0.5224552 , -0.3145284 , -0.07438397], [ 0.6160992 , -0.16827095,  0.44195193, -0.78394   , -0.73066866]])]
    weights = None
    model = get_model(expr, dataset.n_node_features, channels, dataset.n_classes, weights)
    node = None

    # import lirpa_domain as domain_lirpa
    # run_interpreter(dataset, 0.1, domain_lirpa, model, 'node+edge_labels', node)
    # abs_settings = AbstractionSettings(0.1, 0.1, EdgeAbstraction({(i, i) for i in range(dataset[0].n_nodes)}, True, edge_label_generator='GCN'), 'backward')
    # abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(), 'backward')
    abs_settings = AbstractionSettings(0, 0, BisimAbstraction('fw'), 'backward')
    new_run_interpreter(dataset, model, abs_settings, None)

if __name__ == '__main__':
    i = 0
    while i < 100:
        interpreter_debug()
        i = i + 1

