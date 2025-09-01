from mpmath import iv, mp
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import _EagerTensorBase, Operation
# from interval import interval, inf, imath # https://pyinterval.readthedocs.io/en/latest/guide.html
# from intervals.number import Interval as I # https://github.com/marcodeangelis/intervals

mp.dps = 2
mp.pretty = True
iv.dps = 2
iv.pretty = True

# iv.matrix([iv.mpf([0, 2]), iv.mpf([3, 4]), iv.mpf([-2, -1])])

# Basic abstract functions (Forward)
def AReLU(intv: iv.mpf) -> iv.mpf:
    if intv > 0:
        return intv
    elif intv <= 0:
        return iv.mpf(0)
    else:
        return iv.mpf([0, intv.b])


seq_AReLU = np.vectorize(AReLU)
seq_exp = np.vectorize(iv.exp)
left_endpoints = np.vectorize(lambda x: mp.mpf(x.a))
right_endpoints = np.vectorize(lambda x: mp.mpf(x.b))


def softmax_naive(intv_arr: iv.matrix) -> iv.matrix:
    exps = seq_exp(intv_arr)
    return exps / sum(exps)


def softmax_exact(intv_arr: iv.matrix) -> iv.matrix:
    left_exp = seq_exp(left_endpoints(intv_arr))
    right_exp = seq_exp(right_endpoints(intv_arr))
    outputs = []
    for i in range(len(intv_arr)):
        outputs.append(
            iv.mpf([left_exp[i] / (sum(right_exp) - (right_exp[i] - left_exp[i])), right_exp[i] / (sum(left_exp) + (right_exp[i] - left_exp[i]))]))
    return np.array(outputs)

# abstract psi functions (Forward)
def make_layer(w, activation):
    w = iv.matrix(w)
    if activation == 'RELU':
        def lin(X, x):
            return seq_AReLU(x * w)
    else:
        def lin(X, x):
            return softmax_exact(x * w)
    return lin

# abstract phi functions (Forward)
def prod(i, e, j):
    return i * e

# abstract sigma functions (Forward)
def sm(m, x):
    return sum(m)


# Basic abstract functions (Backward)

def BAReLU(intv: iv.mpf) -> iv.mpf:
    if intv > 0:
        return intv
    elif intv == iv.mpf(0):
        return iv.mpf(['-inf', 0])
    else:
        return iv.mpf(['-inf', intv.b])


seq_BAReLU = np.vectorize(BAReLU)


# abstract psi functions (Forward)
def make_layer(w, activation):
    w = iv.matrix(w)
    if activation == 'RELU':
        def lin(X, x):
            return seq_AReLU(x * w)
    else:
        def lin(X, x):
            return softmax_exact(x * w)
    return lin

# abstract phi functions (Forward)
def prod(i, e, j):
    return i * e

# abstract sigma functions (Backward)
# From the labels, obtain the messages
def bsm(x, n_messages):
    return sum(x)



def get_node_labels(x, node_id):
    labels = []
    for m in x:
        labels.append(m[node_id, :])
    return tuple(labels)


def abstract(value: tf.Tensor, delta: float = 0) -> iv.matrix:
    x = value.numpy()
    intv_arr = iv.matrix([[iv.mpf([elem - delta, elem + delta]) for elem in row] for row in x])
    # intv_arr = [[interval[elem - delta, elem + delta] for elem in row] for row in x]
    return intv_arr


def concretize(avalue: iv.matrix) -> tuple[Operation | _EagerTensorBase, ...]:
    output = []
    for j in range(avalue.cols):
        intv_arr = avalue[:, j]
        output.append(tf.constant([[float(intv.a), float(intv.b)] for intv in intv_arr]))
    return tuple(output)


def abs_apply(psi, x):
    n_nodes = x[0].rows
    embeddings = list(map(lambda i: psi(*x, *get_node_labels(x, i)).tolist(), range(n_nodes)))
    return iv.matrix(embeddings)

def abs_pre(phi, sigma, x, a, e):
    n_nodes = x[0].rows
    index_targets = a.indices[:, 1].numpy()  # Nodes receiving the message
    index_sources = a.indices[:, 0].numpy()  # Nodes sending the message (ie neighbors)
    # Message
    messages = [[] for _ in range(n_nodes)]  # list of lists of messages
    for src, label, tgt in zip(index_sources, e, index_targets):
        messages[tgt].append(phi(*get_node_labels(x, src), label, *get_node_labels(x, tgt)))
    # Aggregate
    embeddings = list(map(lambda m: sigma(m[1], *get_node_labels(x, m[0])).tolist()[0], enumerate(messages)))
    # Update
    return iv.matrix(embeddings)



