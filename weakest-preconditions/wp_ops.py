from typing import Callable

import numpy as np
import tensorflow as tf
from z3 import And, ArrayRef, Sum, ExprRef, If


def tensor_to_interval(x: tf.Tensor, delta: float, array_var: ArrayRef) -> ExprRef:
    """Turns a feature matrix into a set of constraints by binding each value in it to its enclosing delta interval.

    :param x: The feature matrix as a 2-dimensional tensor.
    :param delta: The delta to add and subtract to each value.
    :param array_var: The array variable reference generated by z3.
    :return: A z3 conjunction of all the constraints.
    """
    x = x.numpy()
    intv_arr = [elem for i, row in enumerate(x) for j in range(len(row)) for elem in (array_var[i][j] >= float(x[i][j]) - delta, array_var[i][j] <= float(x[i][j]) + delta)]
    return And(intv_arr)


# applies to a single node
def make_layer(w: np.ndarray[tuple[int, int], np.dtype[np.float32]]) -> Callable[[ArrayRef, int, int], tuple[list[list[ExprRef]], int]]:
    """Creates a linear layer function using the given weight matrix.

    :param w: A weight matrix as a 2-dimensional numpy array
    :return: A linear layer function (with no bias).
    """

    def lin(x: ArrayRef, n_nodes: int, n_node_features: int) -> tuple[list[list[ExprRef]], int]:
        """Performs a linear layer computation on the full node feature matrix. (Every node).

        :param x: The array variable reference generated by z3.
        :param n_nodes: Number of nodes in the graph.
        :param n_node_features: Number of features in each node.
        :return: A new node feature matrix of z3 symbolic expressions, and the new number of features (corresponding to the length of each node feature in the
        new matrix).
        """
        out = [[Sum([x[i][k] * w[k][j] for k in range(len(w))]) for j in range(len(w[0]))] for i in range(n_nodes)]
        return out, len(w)

    return lin

def relu(x: ArrayRef, n_nodes: int, n_node_features: int) -> tuple[list[list[ExprRef]], int]:
    """Performs a relu on the full node feature matrix. (Every node).

    :param x: The array variable reference generated by z3.
    :param n_nodes: Number of nodes in the graph.
    :param n_node_features: Number of features in each node.
    :return: A new node feature matrix of z3 symbolic expressions, and the new number of features (corresponding to the length of each node feature in the
    new matrix).
    """
    return [[If(x[i][j] > 0, x[i][j], 0) for j in range(n_node_features)] for i in range(n_nodes)], n_node_features

def idn(x: ArrayRef, n_nodes: int, n_node_features: int) -> tuple[list[list[ExprRef]], int]:
    """Performs the identity function the full node feature matrix. (Every node).

    :param x: The array variable reference generated by z3.
    :param n_nodes: Number of nodes in the graph.
    :param n_node_features: Number of features in each node.
    :return: A new node feature matrix of z3 symbolic expressions, and the new number of features (corresponding to the length of each node feature in the
    new matrix).
    """
    return [[x[i][j] for j in range(n_node_features)] for i in range(n_nodes)], n_node_features

# generates a single message
def product(i: ArrayRef, e: tf.Tensor, j: ArrayRef, n_node_features: int) -> tuple[list[ExprRef], int]:
    """Performs the product of the labels of the sender node with the edge label. Applied to a single node.

    :param i: The (inner) array variable reference generated by z3 for a sender node.
    :param e: The label on the edge between i and j.
    :param j: The (inner) array variable reference generated by z3 for a receiving node.
    :param n_node_features: Number of features in each node.
    :return: The list of messages, and the new number of features.
    """
    return [i[k] * e[0] for k in range(n_node_features)], n_node_features

# aggregates messages for a single node
def summation(m: list[list[ExprRef]], x: ArrayRef, n_node_features: int) -> tuple[list[ExprRef], int]:
    """Adds all the messages. Applied to a single node.

    :param m: The list of messages. Each message is a list of expressions.
    :param x: The (inner) array variable reference generated by z3 for the node receiving these messages.
    :param n_node_features: Number of features in each node.
    :return: A new node label of z3 symbolic expressions, and the new number of features.
    """
    return [Sum(v) for v in zip(*m)], n_node_features

