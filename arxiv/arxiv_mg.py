import numpy as np
import tensorflow as tf
import os
from ogb.nodeproppred import Evaluator, NodePropPredDataset
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, GCNFilter

from libmg import Dataset, Graph, MGCompiler, NodeConfig, EdgeConfig, CompilerConfig, PsiLocal, Phi, Sigma
from scipy.sparse import coo_matrix


class OGBDataset(Dataset):
    def __init__(self, dataset_name, gnn, **kwargs):
        self.te_mask = None
        self.va_mask = None
        self.tr_mask = None
        self.dataset_name = dataset_name
        self.gnn = gnn
        self.n_classes = None
        super().__init__('OGBDataset', **kwargs)

    def read(self):
        filename = os.path.join(self.path, self.dataset_name)
        loaded_graph = np.load(filename + '.npz', allow_pickle=True)
        graph = Graph(x=loaded_graph['x'],
                      a=loaded_graph['a'].item(),
                      e=loaded_graph['e'],
                      y=loaded_graph['y'])
        self.n_classes = loaded_graph['n_classes'].item()
        self.tr_mask = loaded_graph['masks'][0]
        self.va_mask = loaded_graph['masks'][1]
        self.te_mask = loaded_graph['masks'][2]
        return [graph]

    def download(self):
        ogb_dataset = NodePropPredDataset(self.dataset_name)
        transforms = []
        if self.gnn == 'GCN':
            transforms.append(GCNFilter())
        transforms.append(AdjToSpTensor())
        dataset = OGB(ogb_dataset, transforms=transforms)
        # Data splits
        idx = ogb_dataset.get_idx_split()
        idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
        mask_tr = np.zeros(dataset.n_nodes, dtype=bool)
        mask_va = np.zeros(dataset.n_nodes, dtype=bool)
        mask_te = np.zeros(dataset.n_nodes, dtype=bool)
        mask_tr[idx_tr] = True
        mask_va[idx_va] = True
        mask_te[idx_te] = True
        masks = [mask_tr, mask_va, mask_te]
        graph = dataset[0]
        edge_features = tf.expand_dims(tf.cast(graph.a.values, tf.float32), axis=1).numpy()
        new_adj = tf.sparse.map_values(tf.ones_like, graph.a)
        new_adj = coo_matrix((new_adj.values.numpy(), (new_adj.indices[:, 0].numpy(), new_adj.indices[:, 1].numpy())), dtype=np.uint8)
        os.makedirs(self.path)
        filename = os.path.join(self.path, self.dataset_name)
        np.savez(filename, x=graph.x, a=new_adj, e=edge_features, y=graph.y, n_classes=ogb_dataset.num_classes, masks=masks)


if __name__ == '__main__':
    # Load data
    dataset = OGBDataset("ogbn-arxiv", 'GCN')

    # Parameters
    channels = 256  # Number of channels for GCN layers
    dropout = 0.5  # Dropout rate for the features
    learning_rate = 1e-2  # Learning rate
    epochs = 200  # Number of training epochs
    n_nodes = dataset.n_nodes  # Number of nodes in the graph
    n_node_features = dataset.n_node_features  # Original size of node features
    n_edge_features = dataset.n_edge_features
    n_classes = dataset.n_classes  # OGB labels are sparse indices


    expr = '|x>+ ; dense ; |x>+ ; dense ; |x>+ ; out'
    prod = Phi(lambda i, e, j: i * e)
    sm = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))
    dense = PsiLocal.make('dense', tf.keras.layers.Dense(channels, activation='relu', use_bias=False))
    out = PsiLocal.make('out', tf.keras.layers.Dense(n_classes, activation='softmax', use_bias=False))
    config = CompilerConfig.xae_config(NodeConfig(tf.float32, 128), EdgeConfig(tf.float32, 1), tf.uint8, {})
    compiler = MGCompiler({'dense': dense, 'out': out}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    model.summary()

    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SparseCategoricalCrossentropy()


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

    print(tf.argmax(preds, axis=1).numpy())
    print(np.squeeze(y))


