import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.framework.ops import _EagerTensorBase, Operation
from intervals.number import Interval as I # https://github.com/marcodeangelis/intervals

# # Basic abstract functions (Forward)
# def AReLU(x: float) -> float:
#     if x > 0:
#         return x
#     else:
#         return 0


# def seq_AReLU(intv_arr: I) -> I:
#     interval_list = [AReLU(intv) for intv in intv_arr]
#     return I(lo=np.array([intv.lo for intv in interval_list]), hi=np.array([intv.hi for intv in interval_list]))


#
# def softmax_naive(intv_arr: iv.matrix) -> iv.matrix:
#     exps = seq_exp(intv_arr)
#     return exps / sum(exps)
#
#
# def softmax_exact(intv_arr: I) -> I:
#     left_exp = np.exp([intv.lo for intv in intv_arr])
#     right_exp = np.exp([intv.hi for intv in intv_arr])
#     intervals = I(lo=np.array([left_exp[i] / (sum(right_exp) - (right_exp[i] - left_exp[i])) for i in range(len(intv_arr))]), hi=np.array([right_exp[i] / (sum(left_exp) + (right_exp[i] - left_exp[i])) for i in range(len(intv_arr))]))
#     return intervals

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# # abstract psi functions (Forward)
def make_layer(w, activation):
    if activation == 'RELU':
        def lin(X, x):
            return np.maximum(np.matmul(x, w), 0)
    else:
        def lin(X, x):
            return softmax(np.matmul(x, w))
    return lin

# # abstract phi functions (Forward)
def prod(i, e, j):
    return i * e

# # abstract sigma functions (Forward)
def sm(m, x):
    return sum(m)
#
#
# # Basic abstract functions (Backward)
#
# def BAReLU(intv: iv.mpf) -> iv.mpf:
#     if intv > 0:
#         return intv
#     elif intv == iv.mpf(0):
#         return iv.mpf(['-inf', 0])
#     else:
#         return iv.mpf(['-inf', intv.b])
#
#
# seq_BAReLU = np.vectorize(BAReLU)
#
#
# # abstract psi functions (Forward)
# def make_layer(w, activation):
#     w = iv.matrix(w)
#     if activation == 'RELU':
#         def lin(X, x):
#             return seq_AReLU(x * w)
#     else:
#         def lin(X, x):
#             return softmax_exact(x * w)
#     return lin
#
# # abstract phi functions (Forward)
# def prod(i, e, j):
#     return i * e
#
# # abstract sigma functions (Backward)
# # From the labels, obtain the messages
# def bsm(x, n_messages):
#     return sum(x)
#
#
#
def get_node_labels(x, node_id):
    labels = []
    for m in x:
        labels.append(m[node_id, :])
    return tuple(labels)


def abstract(value: tf.Tensor, delta: float = 0) -> np.ndarray:
    x = value.numpy()
    return x


def concretize(avalue: np.ndarray) -> tuple[tf.Tensor, ...]:
    return (tf.convert_to_tensor(avalue), )


def abs_apply(psi, x):
    n_nodes = x[0].shape[0]
    embeddings = list(map(lambda i: psi(*x, *get_node_labels(x, i)), tqdm(range(n_nodes))))
    embeddings = np.array(embeddings)
    return embeddings

def abs_pre(phi, sigma, x, a, e):
    n_nodes = x[0].shape[0]
    index_targets = a.indices[:, 1].numpy()  # Nodes receiving the message
    index_sources = a.indices[:, 0].numpy()  # Nodes sending the message (ie neighbors)
    # Message
    messages = [[] for _ in range(n_nodes)]  # list of lists of messages
    for src, label, tgt in tqdm(zip(index_sources, e, index_targets), total=len(e)):
        messages[tgt].append(phi(*get_node_labels(x, src), label, *get_node_labels(x, tgt)))
    # Aggregate
    embeddings = list(map(lambda m: sigma(m[1], *get_node_labels(x, m[0])), tqdm(enumerate(messages), total=len(messages))))
    embeddings = np.array(embeddings)
    # Update
    return embeddings


if __name__ == '__main__':
    print(abs_pre(make_layer(np.random.rand(3, 5), 'RELU'), (I(lo=np.array([[-2, 1, 2], [1, 2, 3]]), hi=np.array([[2, 3, 5], [4, 5, 6]])),)))