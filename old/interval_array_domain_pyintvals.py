from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import _EagerTensorBase, Operation
from interval import interval, inf, imath # https://pyinterval.readthedocs.io/en/latest/guide.html


# iv.matrix([iv.mpf([0, 2]), iv.mpf([3, 4]), iv.mpf([-2, -1])])

# Basic abstract functions (Forward)
def AReLU(intv: interval) -> interval:
    if intv[0].inf > 0:
        return intv
    elif intv[0].sup <= 0:
        return interval[0]
    else:
        return interval[0, intv[0].sup]



def softmax_exact(intv_arr: list[interval]) -> list[interval]:
    left_exp = np.exp([intv[0].inf for intv in intv_arr])
    right_exp = np.exp([intv[0].sup for intv in intv_arr])
    intervals = [interval[left_exp[i] / (sum(right_exp) - (right_exp[i] - left_exp[i])),
                          right_exp[i] / (sum(left_exp) + (right_exp[i] - left_exp[i]))]
                 for i in range(len(intv_arr))]
    return intervals

# abstract psi functions (Forward)
def make_layer(w, activation):
    if activation == 'RELU':
        def lin(X, x):
            return [AReLU(sum([x[i] * w[i, j] for i in range(w.shape[0])])) for j in range(w.shape[-1])]
    else:
        def lin(X, x):
            return softmax_exact([sum([x[i] * w[i, j] for i in range(w.shape[0])]) for j in range(w.shape[-1])])
    return lin

# abstract phi functions (Forward)
def prod(i, e, j):
    return [i[idx] * e[0] for idx in range(len(i))]

# abstract sigma functions (Forward)
def sm(m, x):
    return [sum([item[i] for item in m]) for i in range(len(m[0]))]


def get_node_labels(x, node_id):
    labels = []
    for m in x:
        labels.append(m[node_id])
    return tuple(labels)


def abstract(value: tf.Tensor, delta: float = 0) -> list[list[interval]]:
    x = value.numpy()
    intv_arr = [[interval[elem - delta, elem + delta] for elem in row]for row in x]
    return intv_arr


def concretize(avalue: list[list[interval]]) -> tuple[Operation | _EagerTensorBase, ...]:
    output = []
    for j in range(len(avalue[0])):
        intv_arr = [item[j] for item in avalue]
        output.append(tf.constant([[float(intv[0].inf), float(intv[0].sup)] for intv in intv_arr]))
    return tuple(output)


def abs_apply(psi, x):
    n_nodes = len(x[0])
    embeddings = list(map(lambda i: psi(*x, *get_node_labels(x, i)), tqdm(range(n_nodes))))
    return embeddings

def abs_pre(phi, sigma, x, a, e):
    n_nodes = len(x[0])
    index_targets = a.indices[:, 1].numpy()  # Nodes receiving the message
    index_sources = a.indices[:, 0].numpy()  # Nodes sending the message (ie neighbors)
    # Message
    messages = [[] for _ in range(n_nodes)]  # list of lists of messages
    for src, label, tgt in tqdm(zip(index_sources, e, index_targets), total=len(e)):
        messages[tgt].append(phi(*get_node_labels(x, src), label, *get_node_labels(x, tgt)))
    # Aggregate
    embeddings = list(map(lambda m: sigma(m[1], *get_node_labels(x, m[0])), tqdm(enumerate(messages), total=len(messages))))
    # Update
    return embeddings



