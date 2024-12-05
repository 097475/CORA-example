import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, register_custom_op
import torch
import torch.nn as nn

from auto_LiRPA.operators import BoundLinear, BoundRelu, Bound
from keras.src.ops import dtype

from auto_LiRPA.operators import Interval

from auto_LiRPA.operators import BoundMul


# # Basic abstract functions (Forward)
# def AReLU(intv: I) -> I:
#     if intv > 0:
#         return intv
#     elif intv <= 0:
#         return I(0)
#     else:
#         return I(0, intv.hi)

# def softmax_exact(intv_arr: I) -> I:
#     left_exp = np.exp([intv.lo for intv in intv_arr])
#     right_exp = np.exp([intv.hi for intv in intv_arr])
#     intervals = I(lo=np.array([left_exp[i] / (sum(right_exp) - (right_exp[i] - left_exp[i])) for i in range(len(intv_arr))]), hi=np.array([right_exp[i] / (sum(left_exp) + (right_exp[i] - left_exp[i])) for i in range(len(intv_arr))]))
#     return intervals
#
# # # abstract psi functions (Forward)
# def make_layer(w, activation):
#     if activation == 'RELU':
#         def lin(X, x):
#             return [AReLU(sum(x * w[:, j])) for j in range(w.shape[-1])]
#     else:
#         def lin(X, x):
#             interval_list = [sum(x * w[:, j]) for j in range(w.shape[-1])]
#             return softmax_exact(I(lo=np.array([intv.lo for intv in interval_list]), hi=np.array([intv.hi for intv in interval_list])))
#     return lin
#

# """ Step 1: Define a `torch.autograd.Function` class to declare and implement the
# computation of the operator. """
# class PsiLinear(torch.autograd.Function):
#     @staticmethod
#     def symbolic(g, X, x, w):
#         """ In this function, define the arguments and attributes of the operator.
#         "custom::PsiLinear" is the name of the new operator, "x" is an argument
#         of the operator, "const_i" is an attribute which stands for "c" in the operator.
#         There can be multiple arguments and attributes. For attribute naming,
#         use a suffix such as "_i" to specify the data type, where "_i" stands for
#         integer, "_t" stands for tensor, "_f" stands for float, etc. """
#         return g.op('custom::PsiLinear', X, x, w_t=w)
#
#     @staticmethod
#     def forward(ctx, X, x, w):
#         """ In this function, implement the computation for the operator, i.e.,
#         f(x) = x + c in this case. """
#         return torch.nn.ReLU()(torch.matmul(x, w))


def phi_product(i, e, j):
    return i * e

def abs_phi_product(i, e, j):
    # x is constant
    op = lambda x, const: x * const
    const = e
    inp_lb = i[0]
    inp_ub = i[1]
    pos_mask = (const > 0).to(dtype=inp_lb.dtype)
    neg_mask = 1. - pos_mask
    lb = op(inp_lb, const * pos_mask) + op(inp_ub, const * neg_mask)
    ub = op(inp_ub, const * pos_mask) + op(inp_lb, const * neg_mask)
    return Interval(lb, ub, i.ptb)

def sigma_sum(m, x):
    return torch.stack(m).sum(dim=0)

def abs_sigma_sum(m, x):
    lbs = [msg[0] for msg in m]
    ubs = [msg[1] for msg in m]
    return Interval(lb=sum(lbs), ub=sum(ubs), ptb=x.ptb)


# """ Step 1: Define a `torch.autograd.Function` class to declare and implement the
# computation of the operator. """
class Pre(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e):
        """ In this function, define the arguments and attributes of the operator.
        "custom::SigmaSum" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Pre', x, a, e).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = i * e in this case. """
        n_nodes = x.shape[0]
        n_edges = e.shape[0]
        index_targets = a[1]  # Nodes receiving the message
        index_sources = a[0]  # Nodes sending the message (ie neighbors)
        # Message
        messages = [[] for _ in range(n_nodes)]  # list of lists of messages
        for idx in range(n_edges):
            messages[index_targets[idx]].append(phi( x[index_sources[idx], :], e[idx], x[index_targets[idx], :]))
        # Aggregate
        embeddings = [sigma(m, x[i, :]) for i,m in enumerate(messages)]
        embeddings = torch.stack(embeddings)
        # Update
        return embeddings




def get_node_labels(x, node_id):
    lb = x[0][node_id, :]
    ub = x[1][node_id, :]
    ptb = x.ptb
    return Interval(lb, ub, ptb)




#
# def abstract(value: tf.Tensor, delta: float = 0) -> I:
#     x = value.numpy()
#     intv_arr = I(lo=np.array([[elem - delta for elem in row] for row in x]), hi=np.array([[elem + delta for elem in row] for row in x]))
#     return intv_arr
#
#
# def concretize(avalue: I) -> tuple[Operation | _EagerTensorBase, ...]:
#     output = []
#     for j in range(avalue.shape[-1]):
#         intv_arr = avalue[:, j]
#         output.append(tf.constant([[intv.lo.item(), intv.hi.item()] for intv in intv_arr]))
#     return tuple(output)
#
#
# def abs_apply(psi, x):
#     n_nodes = x[0].shape[0]
#     embeddings = list(map(lambda i: psi(*x, *get_node_labels(x, i)), tqdm(range(n_nodes))))
#     embeddings = I(lo=np.array([[intv.lo for intv in row] for row in embeddings]), hi=np.array([[intv.hi for intv in row] for row in embeddings]))
#     return embeddings
#

# class Psi(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#
#     def forward(self, x):
#         n_nodes = x[0].shape[0]
#         embeddings = list(map(lambda i: self.linear(*get_node_labels(x, i)), range(n_nodes)))
#         embeddings = I(lo=np.array([[intv.lo for intv in row] for row in embeddings]), hi=np.array([[intv.hi for intv in row] for row in embeddings]))
#         return embeddings

""" Step 2: Define a `torch.nn.Module` class to declare a module using the defined
custom operator. """
class PreImage(nn.Module):
    def __init__(self, phi_f, sigma_f):
        super().__init__()
        global phi, sigma, abs_phi, abs_sigma
        phi = phi_f
        abs_phi = abs_phi_product
        sigma = sigma_f
        abs_sigma = abs_sigma_sum

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Pre.apply(x, a, e)


""" Step 3: Implement a Bound class to support bound computation for the new operator. """
class BoundPre(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x, a, e):
        n_nodes = x.shape[0]
        n_edges = e.shape[0]
        index_targets = a[1]  # Nodes receiving the message
        index_sources = a[0]  # Nodes sending the message (ie neighbors)
        # Message
        messages = [[] for _ in range(n_nodes)]  # list of lists of messages
        for idx in range(n_edges):
            messages[index_targets[idx]].append(phi( x[index_sources[idx], :], e[idx], x[index_targets[idx], :]))
        # Aggregate
        embeddings = [sigma(m, x[i, :]) for i,m in enumerate(messages)]
        embeddings = torch.stack(embeddings)
        # Update
        return embeddings

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        print('Calling bound_backward for custom::PlusConstant')
        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self.broadcast_backward(last_A, w)

        e = e.forward_value
        a = a.forward_value
        n_nodes = x.output_shape[0]
        n_edges = e.shape[0]
        index_targets = a[1]  # Nodes receiving the message
        index_sources = a[0]  # Nodes sending the message (ie neighbors)
        # shape = (out_dim, batch_size, in_dim)

        # Manage addition aggregation

        op = lambda i, e, j: i * e
        # Handle the case of multiplication by a constant.
        # Message
        messages = [[] for _ in range(n_nodes)]  # list of lists of messages
        for idx in range(n_edges):
            lAx = (None if last_lA is None
                   else self.broadcast_backward(op(last_lA[:, index_sources[idx]:index_sources[idx]+1, :], e[idx], last_lA[:, index_targets[idx]:index_targets[idx]+1, :]), x))
            uAx = (None if last_uA is None
                   else self.broadcast_backward(op(last_uA[:, index_sources[idx]:index_sources[idx]+1, :], e[idx], last_uA[:, index_targets[idx]:index_targets[idx]+1, :]), x))
            messages[index_targets[idx]].append((lAx, uAx))
        embeddings = [abs_sigma(m, get_node_labels(x, i)) for i,m in enumerate(messages)]
        return [(lAx, uAx), (None, None), (None, None)], 0., 0.

    def interval_propagate(self, *v):
        """ IBP computation """
        print('Calling interval_propagate for custom::Pre')
        x, a, e = v
        e = e[0] # not an interval
        a = a[0] # not an interval
        n_nodes = x[0].shape[0]
        n_edges = e.shape[0]
        index_targets = a[1]  # Nodes receiving the message
        index_sources = a[0]  # Nodes sending the message (ie neighbors)
        # Message
        messages = [[] for _ in range(n_nodes)]  # list of lists of messages
        for idx in range(n_edges):
            messages[index_targets[idx]].append(abs_phi(get_node_labels(x, index_sources[idx]), e[idx], get_node_labels(x, index_targets[idx])))
        # Aggregate
        embeddings = [abs_sigma(m, get_node_labels(x, i)) for i,m in enumerate(messages)]
        lb, ub = torch.stack([emb[0] for emb in embeddings]), torch.stack([emb[1] for emb in embeddings])
        # Update
        return Interval(lb, ub, x.ptb)


# Define computation as a nn.Module.
# class MyModel(nn.Module):
#     def __init__(self, w, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.lin = torch.nn.Linear(3, 3, bias=False)
#         self.lin.weight = nn.Parameter(w)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         # Define your computation here.
#         return self.relu(self.lin(x))


register_custom_op("custom::Pre", BoundPre)

class MyModel2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre = PreImage(phi_product, sigma_sum)

    def forward(self, x, a, e):
        return self.pre(x, a, e)
        # Define your computation here.


class MyModel3(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, a, e):
        return torch.matmul(a, x)
        # return x[0, :] * e[:1, :] + x[1, :] * e[1:2, :] + x[2, :] * e[2:3, :]
        # Define your computation here.


if __name__ == '__main__':
    expr = '<x|+ ; lin ; <x|+ ; out'
    channels = 5

    x_in = torch.tensor([[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]], dtype=torch.float32)
    a_in = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], [0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 1, 4]])
    w_in = torch.tensor([[0.2, 0.1, 0.4, 0, 0], [0, 0.1, 0.2, 0.3, 0], [0, 0.2, 0.5, 0.7, 0], [0, 0, 0, 0.4, 0.5], [0, 0.1, 0, 0, 0.5]])
    e_in = torch.tensor([[0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.40824828], [0.3333333], [0.3333333], [0.40824828], [0.49999997],
                    [0.49999997], [0.40824828], [0.49999997]], dtype=torch.float32)

    model = MyModel3()
    lirpa_model = BoundedModule(model, (torch.empty_like(x_in), w_in, torch.empty_like(e_in)), device=x_in.device, verbose=True)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    my_input = BoundedTensor(x_in, ptb)
    prediction = model(my_input, w_in, e_in)

    print(prediction)

    abs_prediction_2 = lirpa_model.compute_bounds(x=(my_input, w_in, e_in))

    print(abs_prediction_2)


    # model = MyModel2()
    # lirpa_model = BoundedModule(model, (torch.empty_like(x_in), a_in, torch.empty_like(e_in)), device=x_in.device)
    #
    # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # my_input = BoundedTensor(x_in, ptb)
    #
    # prediction = model(my_input, a_in, e_in)
    #
    # print(prediction)
    #
    # abs_prediction = lirpa_model.forward((my_input, a_in, e_in))
    #
    # print(abs_prediction)
    #
    # abs_prediction_2 = lirpa_model.compute_bounds(x=(my_input, a_in, e_in))
    #
    # print(abs_prediction_2)

    # w = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [1, 1, 1]], dtype=torch.float32)
    #
    #
    # model = MyModel(w)
    # my_input = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    # # Wrap the model with auto_LiRPA.
    # model = BoundedModule(model, (torch.empty_like(my_input),))
    # # Define perturbation. Here we add Linf perturbation to input data.
    # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # # Make the input a BoundedTensor with the pre-defined perturbation.
    # my_input = BoundedTensor(my_input, ptb)
    # # Regular forward propagation using BoundedTensor works as usual.
    # prediction = model(my_input)
    # # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
    # lb, ub = model.compute_bounds(x=(my_input,), method="IBP")

    # input_bound, _, _ = my_input.ptb.init(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32))
    # linear = BoundLinear()
    # relu = BoundRelu(attr={}, options={})
    # linear.interval_propagate(input_bound.lower, input_bound.upper, w=w)





