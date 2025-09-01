from typing import Tuple, Optional

import numpy as np
import scipy.sparse
import tensorflow as tf
from lark import v_args
from lark.visitors import Interpreter
from libmg import mg_reconstructor, mg_parser
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, register_custom_op
import torch
import torch.nn as nn
from auto_LiRPA.operators import Bound, Interval, BoundLinear, MulHelper, BoundMul, multiply_by_A_signs, BoundInput, BoundParams
from libmg.compiler.compiler import Context
from libmg.compiler.layers import FunctionApplication
from keras import layers
from torch import Tensor
from tqdm import tqdm

from auto_LiRPA.operators import BoundMatMul

from auto_LiRPA.operators import BoundTranspose


# def check_soundness(pred, lb, ub):
#     eps = 0.000001
#     pred = pred[0] if isinstance(pred, tuple) else pred
#     pred = pred[0] if pred.shape.ndims == 3 else pred
#     lb = lb[0] if lb.ndim == 3 else lb
#     ub = ub[0] if ub.ndim == 3 else ub
#     for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
#         for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
#             assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
#     print('Soundness check passed')


# def check_soundness(pred, lb, ub):
#     eps = 0.000001
#     pred = pred[0] if isinstance(pred, tuple) else pred
#     pred = pred[0] if pred.shape.ndims == 3 else pred
#     lb = lb[0] if lb.ndim == 3 else lb
#     ub = ub[0] if ub.ndim == 3 else ub
#     for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
#         for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
#             assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
#     print('Soundness check passed')

# def forward_lub(ml1, mu1, l1, u1, ml2, mu2, l2, u2):
#     mask = torch.flatten(l1[0]) <= torch.flatten(l2[0])
#     expanded_mask = mask[:, None]
#     ml1 = torch.transpose(torch.reshape(ml1, (1, 15, -1,)), 1, 2)
#     ml2 = torch.transpose(torch.reshape(ml2, (1, 15, -1,)), 1, 2)
#     lb = torch.where(expanded_mask, ml1[0], ml2[0])
#     mask = torch.flatten(u1[0]) >= torch.flatten(u2[0])
#     expanded_mask = mask[:, None]
#     mu1 = torch.transpose(torch.reshape(mu1, (1, 15, -1,)), 1, 2)
#     mu2 = torch.transpose(torch.reshape(mu2, (1, 15, -1,)), 1, 2)
#     ub = torch.where(expanded_mask, mu1[0], mu2[0])
#     return torch.reshape(torch.transpose(lb, 0, 1), (1, 15, 5, 3)), torch.reshape(torch.transpose(ub, 0, 1), (1, 15, 5, 3))

# Helper functions
def get_node_labels(x, node_id):
    lb = x[0][0][node_id, :]
    ub = x[1][0][node_id, :]
    return lb, ub

def get_edge_labels(e, edge_id):
    lb = e[0][edge_id, :]
    ub = e[1][edge_id, :]
    return lb, ub

def interval_to_bounded_tensor(lb, ub):
    ptb = PerturbationLpNorm(norm=np.inf, x_L=lb, x_U=ub)
    return BoundedTensor(torch.zeros_like(lb), ptb)

# Phi functions
def phi_product(i, e, j):
    return i * e


# def abs_phi_product(i: Tuple[torch.Tensor, torch.Tensor], e: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], j: Tuple[torch.Tensor, torch.Tensor]):
#     if isinstance(e, tuple):
#         x, y = i, e
#         r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
#         lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
#         upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
#         return lower, upper
#     else:
#         op = lambda x, const: x * const
#         const = e
#         inp_lb = i[0]
#         inp_ub = i[1]
#         pos_mask = (const > 0).to(dtype=inp_lb.dtype)
#         neg_mask = 1. - pos_mask
#         lb = op(inp_lb, const * pos_mask) + op(inp_ub, const * neg_mask)
#         ub = op(inp_ub, const * pos_mask) + op(inp_lb, const * neg_mask)
#         return lb, ub


def abs_phi_product(i: Tuple[torch.Tensor, torch.Tensor], e: Tuple[torch.Tensor, torch.Tensor], j: Tuple[torch.Tensor, torch.Tensor]):
    x, y = i, e
    r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
    lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
    upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
    return lower, upper
# def abs_phi_product2(i, e, j):
#     x, y = i, e
#     r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
#     lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
#     upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
#     return lower, upper

# def abs_fwd_phi_product(i, e, j):
#     return i * e

# def abs_phi_prod2(e):
#     elow, ehigh, xlow, xhigh = e
#
#     if elow is None and ehigh is None:
#         return torch.zeros_like(xlow), torch.zeros_like(xhigh)
#
#     low_edge = []
#     high_edge = []
#     if (elow * xlow < ehigh * xlow and elow * xlow < ehigh * xhigh) or (elow * xhigh < ehigh * xhigh and elow * xhigh < ehigh * xlow):
#         low_edge.append(elow)
#         high_edge.append(ehigh)
#     else:
#         low_edge.append(ehigh)
#         high_edge.append(elow)
#     lowt = torch.stack(low_edge).squeeze()
#     hight = torch.stack(high_edge).squeeze()
#     return lowt, hight

# def abs_bwd_phi_product(e):
#     elow, ehigh, xlow, xhigh = e
#
#     if elow is None and ehigh is None:
#         return torch.zeros_like(xlow), torch.zeros_like(xhigh)
#
#     low_edge = []
#     high_edge = []
#     for k in range(len(xlow)):
#         if (elow * xlow[k] < ehigh * xlow[k] and elow * xlow[k] < ehigh * xhigh[k]) or (elow * xhigh[k] < ehigh * xhigh[k] and elow * xhigh[k] < ehigh * xlow[k]):
#             low_edge.append(elow)
#             high_edge.append(ehigh)
#         else:
#             low_edge.append(ehigh)
#             high_edge.append(elow)
#     lowt = torch.stack(low_edge).squeeze()
#     hight = torch.stack(high_edge).squeeze()
#     return lowt, hight


# def abs_bwd_phi_product(i, e, j):
#     ilow, ihigh = i
#     elow, ehigh, xlow, xhigh = e
#
#     if elow is None and ehigh is None:
#         return torch.zeros_like(ilow), torch.zeros_like(ihigh)
#
#     low_edge = []
#     high_edge = []
#     for k in range(len(ilow)):
#         if (elow * xlow[k] < ehigh * xlow[k] and elow * xlow[k] < ehigh * xhigh[k]) or (elow * xhigh[k] < ehigh * xhigh[k] and elow * xhigh[k] < ehigh * xlow[k]):
#             low_edge.append(elow)
#             high_edge.append(ehigh)
#         else:
#             low_edge.append(ehigh)
#             high_edge.append(elow)
#     lowt = torch.stack(low_edge).squeeze()
#     hight = torch.stack(high_edge).squeeze()
#     return ilow*lowt, ihigh*hight


# def abs_bwd_phi_product_with_bot(i, e, j):
#     ilow, ihigh = i
#     elow, ehigh, xlow, xhigh = e
#     bot = torch.zeros_like(elow)
#
#     low_edge = []
#     high_edge = []
#     edge_values = [elow, elow, ehigh, ehigh, bot, bot]
#     for k in range(len(ilow)):
#         comps = np.array([elow * xlow[k], elow * xhigh[k], ehigh * xlow[k], ehigh * xhigh[k], bot * xlow[k], bot * xhigh[k]])
#         min_index, max_index = np.argmin(comps), np.argmax(comps)
#         low_edge.append(edge_values[min_index])
#         high_edge.append(edge_values[max_index])
#     lowt = torch.stack(low_edge).squeeze()
#     hight = torch.stack(high_edge).squeeze()
#     return ilow*lowt, ihigh*hight



# def abs_bwd_phi_product2(i, e, j):
#     ilow, ihigh = i
#     elow, ehigh, xlow, xhigh = e
#
#     if elow is None and ehigh is None:
#         return torch.zeros_like(ilow), torch.zeros_like(ihigh)
#
#     low_edge = []
#     high_edge = []
#     if (elow * xlow < ehigh * xlow and elow * xlow < ehigh * xhigh) or (elow * xhigh < ehigh * xhigh and elow * xhigh < ehigh * xlow):
#         low_edge.append(elow)
#         high_edge.append(ehigh)
#     else:
#         low_edge.append(ehigh)
#         high_edge.append(elow)
#     lowt = torch.stack(low_edge).squeeze()
#     hight = torch.stack(high_edge).squeeze()
#     return ilow*lowt, ihigh*hight


# Sigma functions
def sigma_sum(m, x):
    return torch.stack(m).sum(dim=0)

def abs_sigma_sum(m, x):
    lbs = [msg[0] for msg in m]
    ubs = [msg[1] for msg in m]
    return sum(lbs), sum(ubs)

# def abs_fwd_sigma_sum(m, x):
#     return sum(m)

def abs_bwd_sigma_sum(m, x):
    return sum(m)


# Psi functions
def make_layer(w, bias, activation):
    in_features = w.shape[0]
    out_features = w.shape[1]
    lin = torch.nn.Linear(in_features, out_features, bias=True)
    lin.weight.data = torch.tensor(w).transpose(0, 1)
    lin.bias.data = torch.tensor(bias)
    if activation == 'relu':
        act = torch.nn.ReLU()
    else:
        act = torch.nn.Identity()
    return lin, act

class Pool(nn.Module):
    def __init__(self, pool_fn, abstraction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_fn = pool_fn
        self.abstraction = abstraction

    def forward(self, x):
        x1 = self.abstraction.handle_pooling(x)
        x2 = self.pool_fn(x1, dim=1, keepdim=True)
        return x2

def make_pooling(pool, abstraction):
    pool_fn = torch.sum if pool == 'sum' else torch.mean
    return Pool(pool_fn, abstraction)


# Least upper bound
def least_upper_bound(lt1, ut1, lt2, ut2):
    olb = torch.minimum(lt1, lt2)
    oub = torch.maximum(ut1, ut2)
    return olb, oub

def backward_lub(ml1, mu1, l1, u1, ml2, mu2, l2, u2):
    mask = torch.flatten(l1[0]) <= torch.flatten(l2[0])
    expanded_mask = mask[:, None, None]
    lb = torch.where(expanded_mask, ml1[0], ml2[0])
    mask = torch.flatten(u1[0]) >= torch.flatten(u2[0])
    expanded_mask = mask[:, None, None]
    ub = torch.where(expanded_mask, mu1[0], mu2[0])
    return lb.unsqueeze(0), ub.unsqueeze(0)


# Abstraction functions
def abstract_x(value: np.ndarray, delta: float = 0, x_L:np.ndarray = None, x_U:np.ndarray = None) -> BoundedTensor | torch.Tensor:
    tensor = torch.tensor(value).unsqueeze(0)
    x_L = torch.tensor(x_L).unsqueeze(0) if x_L is not None else x_L
    x_U = torch.tensor(x_U).unsqueeze(0) if x_U is not None else x_U
    if delta != 0 and x_L is not None and x_U is not None:
        x_L = x_L - delta
        x_U = x_U + delta
        perturbation = PerturbationLpNorm(x_L=x_L, x_U=x_U)
        tensor = BoundedTensor(tensor, perturbation)
    elif delta != 0 or (x_L is not None and x_U is not None):
        perturbation = PerturbationLpNorm(eps=delta, x_L=x_L, x_U=x_U)
        tensor = BoundedTensor(tensor, perturbation)
    return tensor


def abstract_e(value: np.ndarray, delta: float = 0, x_L:np.ndarray = None, x_U:np.ndarray = None) -> BoundedTensor | torch.Tensor:
    tensor = torch.tensor(value)
    x_L = torch.tensor(x_L) if x_L is not None else x_L
    x_U = torch.tensor(x_U) if x_U is not None else x_U
    if delta != 0 and x_L is not None and x_U is not None:
        x_L = x_L - delta
        x_U = x_U + delta
        perturbation = PerturbationLpNorm(x_L=x_L, x_U=x_U)
        tensor = BoundedTensor(tensor, perturbation)
    elif delta != 0 or (x_L is not None and x_U is not None):
        perturbation = PerturbationLpNorm(eps=delta, x_L=x_L, x_U=x_U)
        tensor = BoundedTensor(tensor, perturbation)
    return tensor

def abstract_adj(mat: scipy.sparse.coo_matrix) -> torch.Tensor:
    rows = mat.row
    cols = mat.col
    data = mat.data.astype(np.int32)
    tensor = torch.tensor(np.array([[rows, cols, data]]))
    return tensor


# Concretization
def run_abstract_model(model, x, a, e, algorithm):
    lirpa_model = BoundedModule(model, (torch.empty_like(x), a, torch.empty_like(e)), device=a.device, verbose=True)
    lb, ub = lirpa_model.compute_bounds(x=(x, a, e), method=algorithm, IBP=True)
    print(lirpa_model.save_intermediate())
    return lb.detach()[0], ub.detach()[0]

# Function store
class Transformer:
    def __init__(self, concrete_transformer, interval_transformer, id_element):
        self.concrete_transformer = concrete_transformer
        self.interval_transformer = interval_transformer
        self.id_element = id_element

    def identity_like(self, t):
        tensor = torch.full_like(t, self.id_element)
        return tensor, tensor

fucts = {'x': Transformer(phi_product, abs_phi_product, 1.),
         '+': Transformer(sigma_sum, abs_sigma_sum, 0.)}

# Message-passing procedures
def concrete_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, x, a, e):
    if use_optimized_gcn:
        return concrete_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)
    else:
        return concrete_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)


@torch.no_grad()
def concrete_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    phi = function_store_phi.concrete_transformer
    sigma = function_store_sigma.concrete_transformer
    n_nodes = x.shape[1]
    n_edges = e.shape[0]
    index_targets = a[0][tgt_idx]  # Nodes receiving the message
    index_sources = a[0][src_idx]  # Nodes sending the message (ie neighbors)
    x = x[0]
    # e = e[0]
    # Message
    messages = [[] for _ in range(n_nodes)]  # list of lists of messages
    for idx in range(n_edges):
        messages[index_targets[idx]].append(phi(x[index_sources[idx], :], e[idx], x[index_targets[idx], :]))
    # Aggregate
    embeddings = [sigma(m, x[i, :]) for i, m in enumerate(messages)]
    embeddings = torch.stack(embeddings).unsqueeze(0)
    # Update
    return embeddings

@torch.no_grad()
def concrete_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    x = x[0]
    # e = e[0]
    # Message
    return torch.matmul(e.transpose(-1, -2) if src_idx == 0 else e, x)
    # messages = [[] for _ in range(n_nodes)]  # list of lists of messages
    # for idx in range(n_edges):
    #     messages[index_targets[idx]].append(phi(x[index_sources[idx], :], e[idx], x[index_targets[idx], :]))
    # # Aggregate
    # embeddings = [sigma(m, x[i, :]) for i, m in enumerate(messages)]
    # embeddings = torch.stack(embeddings).unsqueeze(0)
    # # Update
    # return embeddings

def interval_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, x, a, e):
    if use_optimized_gcn:
        return interval_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)
    else:
        return interval_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)

@torch.no_grad()
def interval_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    abs_phi = function_store_phi.interval_transformer
    abs_sigma = function_store_sigma.interval_transformer
    a = a[0][0]  # not an interval
    n_nodes = x[0].shape[1]
    index_targets = a[tgt_idx]  # Nodes receiving the message
    index_sources = a[src_idx]  # Nodes sending the message (ie neighbors)
    edge_status = a[-1]
    messages = [[] for _ in range(n_nodes)]  # list of lists of messages

    n_edges = e[0].shape[0]
    # Message
    for idx in range(n_edges):
        certain_edge = True if edge_status[idx] == 1 else False
        if certain_edge is True:
            messages[index_targets[idx]].append(abs_phi(get_node_labels(x, index_sources[idx]), get_edge_labels(e, idx), get_node_labels(x, index_targets[idx])))
        else:
            msg = abs_phi(get_node_labels(x, index_sources[idx]), get_edge_labels(e, idx) , get_node_labels(x, index_targets[idx]))
            bot = function_store_sigma.identity_like(get_node_labels(x, index_sources[idx])[0]) # abs_phi(get_node_labels(x, index_sources[idx]), None, get_node_labels(x, index_targets[idx]))
            lub = least_upper_bound(*msg, *bot)
            messages[index_targets[idx]].append(lub)

    # Aggregate
    embeddings = [abs_sigma(m, get_node_labels(x, i)) for i, m in enumerate(messages)]
    lb, ub = torch.stack([emb[0] for emb in embeddings]), torch.stack([emb[1] for emb in embeddings])
    # Update
    return torch.unsqueeze(lb, 0), torch.unsqueeze(ub, 0)

@torch.no_grad()
def interval_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    abs_phi = function_store_phi.interval_transformer
    abs_sigma = function_store_sigma.interval_transformer
    # a = a[0][0]  # not an interval
    n_nodes = x[0].shape[1]
    index_targets = a[tgt_idx]  # Nodes receiving the message
    index_sources = a[src_idx]  # Nodes sending the message (ie neighbors)
    edge_status = a[0] #torch.zeros_like(e[0]) # e[0].clone()
    # edge_status[0][a[0], a[1]] = a[2].float()

    # messages = [[] for _ in range(n_nodes)]  # list of lists of messages

    v = [e, x]

    # This will convert an Interval object to tuple.
    # We need to add perturbation property later.
    v_lb, v_ub = zip(*v)
    v_lb = list(v_lb)
    v_ub = list(v_ub)
    v_lb[0] = v_lb[0].transpose(-2, -1) if src_idx == 0 else v_lb[0]

    # vlb0 = v_lb[0].clone()
    # vlb0[edge_status < 0] = torch.clamp(v_lb[0][edge_status < 0], max=0)
    # v_lb[0] = vlb0

    v_lb[1] = v_lb[1].transpose(-2, -1)
    v_ub[0] = v_ub[0].transpose(-2, -1) if src_idx == 0 else v_ub[0]

    # vub0 = v_ub[0].clone()
    # vub0[edge_status < 0] = torch.clamp(v_ub[0][edge_status < 0], min=0)
    # v_ub[0] = vub0

    v_ub[1] = v_ub[1].transpose(-2, -1)
    # After preprocess the lower and upper bounds, we make them Intervals again.
    v = [Interval.make_interval(bounds[0], bounds[1], bounds[2])
         for bounds in zip(v_lb, v_ub, v)]

    x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
    y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
    # Reuse the multiplication bounds and sum over results.
    lower, upper = BoundMul.interval_propagate_both_perturbed(*[(x_l, x_u), (y_l, y_u)])
    lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)


    l, u = lower, upper
    return l, u


#old version, noabstraction
# @torch.no_grad()
# def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     abs_bwd_phi = function_store_phi.poly_transformer_bwd
#     abs_bwd_sigma = function_store_sigma.poly_transformer_bwd
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = x.output_shape[1] * x.output_shape[2]
#     n_node_features_pre = last_lA.shape[-1]
#     n_edges = e.output_shape[0]
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     elow, ehigh = e.interval[0], e.interval[1]
#     xlow, xhigh = x.interval[0][0], x.interval[1][0]
#
#
#     lA = torch.reshape(torch.eye(n_vars_pre), (n_vars_pre, 1, n_nodes, n_node_features_pre))
#     uA = torch.reshape(torch.eye(n_vars_pre), (n_vars_pre, 1, n_nodes, n_node_features_pre))
#     messages_low = [[] for _ in range(n_nodes)]
#     messages_high = [[] for _ in range(n_nodes)]# list of lists of messages
#     for idx in range(n_edges):
#         sidx = index_sources[idx]
#         tidx = index_targets[idx]
#         msg_low, msg_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#                                                         (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#                                                         (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#         messages_low[index_targets[idx]].append(msg_low)
#         messages_high[index_targets[idx]].append(msg_high)
#
#     embeddings_low = torch.concat(
#         [abs_bwd_sigma(m, lA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_low)])
#
#     embeddings_high = torch.concat(
#         [abs_bwd_sigma(m, uA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_high)])
#
#     lAx = embeddings_low
#     uAx = embeddings_high
#
#     reshaped_last_la = torch.reshape(last_lA, (n_vars_post, 1, -1))
#     reshaped_lax = torch.reshape(lAx, (n_vars_pre, 1, -1))
#     lA = torch.zeros_like(last_lA)
#     reshaped_last_ua = torch.reshape(last_uA, (n_vars_post, 1, -1))
#     reshaped_uax = torch.reshape(uAx, (n_vars_pre, 1, -1))
#     uA = torch.zeros_like(last_uA)
#     for v in range(n_vars_post):
#         print(v, 'out of', n_vars_post)
#         non_zero_indices = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#         if non_zero_indices.nelement() != 0:
#             prev_values_la = reshaped_last_la[v][0][non_zero_indices]
#             prev_values_ua = reshaped_last_ua[v][0][non_zero_indices]
#             values_la = torch.unbind(reshaped_lax[non_zero_indices])
#             values_ua = torch.unbind(reshaped_uax[non_zero_indices])
#             rec_value_la = reshaped_last_la[v]
#             rec_value_ua = reshaped_last_ua[v]
#             messages = []
#             for coeff_la, mat_la, coeff_ua, mat_ua in zip(prev_values_la, values_la, prev_values_ua, values_ua):
#                 msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                 messages.append((msg_low, msg_high))
#             embeddings_low, embeddings_high = abs_sigma(messages, (rec_value_la, rec_value_ua))
#         else:
#             embeddings_low = torch.zeros_like(reshaped_lax[0])
#             embeddings_high = torch.zeros_like(reshaped_uax[0])
#         lA[v] = torch.reshape(embeddings_low, (1, n_nodes, n_node_features_pre))#.detach()
#         uA[v] = torch.reshape(embeddings_high, (1, n_nodes, n_node_features_pre))
#
#     lAx = lA
#     uAx = uA
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.




# old version, edge_abs + no_abs, edge postimg faulty
# @torch.no_grad()
# @profile
# def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     abs_bwd_phi = abs_bwd_phi_product
#     abs_bwd_sigma = abs_bwd_sigma_sum
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
#     n_node_features_pre = last_lA.shape[-1]
#     n_edges = e.output_shape[0]
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     edge_status = a.forward_value[0][-1]
#     elow, ehigh = e.interval[0], e.interval[1]
#     xlow, xhigh = x.interval[0][0], x.interval[1][0]
#
#
#     lA = torch.eye(n_vars_pre).view((n_vars_pre, 1, n_nodes, n_node_features_pre))
#     uA = lA
#
#     messages_low = [[] for _ in range(n_nodes)]
#     messages_high = [[] for _ in range(n_nodes)]# list of lists of messages
#     for idx in range(n_edges):
#         certain_edge = True if edge_status[idx] == 1 else False
#         sidx = index_sources[idx]
#         tidx = index_targets[idx]
#         if certain_edge is True:
#             msg_low, msg_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#                                                             (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#                                                             (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#         else:
#             msg_low, msg_high = abs_bwd_phi_product_with_bot((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#                                             (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#                                             (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # m_low, m_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#             #                                 (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#             #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # bot_low, bot_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#             #                                 (None, None, xlow[sidx], xhigh[sidx]),
#             #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # msg_low, msg_high = least_upper_bound(m_low, m_high, bot_low, bot_high)
#         messages_low[index_targets[idx]].append(msg_low)
#         messages_high[index_targets[idx]].append(msg_high)
#
#     embeddings_low = torch.concat(
#         [abs_bwd_sigma(m, lA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_low)])
#
#     embeddings_high = torch.concat(
#         [abs_bwd_sigma(m, uA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_high)])
#
#     lAx = embeddings_low
#     uAx = embeddings_high
#
#     reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
#     reshaped_lax = lAx.view((n_vars_pre, 1, -1))
#     reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
#     reshaped_uax = uAx.view((n_vars_pre, 1, -1))
#
#     lA, uA = [], []
#     for v in range(n_vars_post):
#         print(v, 'out of', n_vars_post)
#         non_zero_indices = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#         if non_zero_indices.nelement() != 0:
#             prev_values_la = reshaped_last_la[v][0][non_zero_indices]
#             prev_values_ua = reshaped_last_ua[v][0][non_zero_indices]
#             values_la = torch.unbind(reshaped_lax[non_zero_indices])
#             values_ua = torch.unbind(reshaped_uax[non_zero_indices])
#             rec_value_la = reshaped_last_la[v]
#             rec_value_ua = reshaped_last_ua[v]
#             messages = []
#             for coeff_la, mat_la, coeff_ua, mat_ua in zip(prev_values_la, values_la, prev_values_ua, values_ua):
#                 msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                 # msg_low, msg_high = mat_la * coeff_la, mat_ua * coeff_ua
#                 messages.append((msg_low, msg_high))
#             embeddings_low, embeddings_high = abs_sigma(messages, (rec_value_la, rec_value_ua))
#         else:
#             embeddings_low = torch.zeros_like(reshaped_lax[0])
#             embeddings_high = torch.zeros_like(reshaped_uax[0])
#         lA.append(embeddings_low)
#         uA.append(embeddings_high)
#
#     lAx = torch.vstack(lA).view((len(lA), 1, n_nodes, n_node_features_pre))
#     uAx = torch.vstack(uA).view((len(uA), 1, n_nodes, n_node_features_pre))
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.

# @torch.no_grad()
# @profile
# def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     # abs_bwd_phi = function_store_phi.poly_transformer_bwd
#     # abs_bwd_sigma = function_store_sigma.poly_transformer_bwd
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
#     n_node_features_pre = last_lA.shape[-1]
#     n_node_features_post = n_vars_post // n_nodes
#     n_edges = e.output_shape[0]
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     edge_status = a.forward_value[0][-1]
#     elow, ehigh = e.interval[0], e.interval[1]
#     xlow, xhigh = x.interval[0][0], x.interval[1][0]
#
#
#     lA = torch.eye(n_vars_pre).view((n_vars_pre, 1, n_nodes, n_node_features_pre))
#     uA = lA
#
#     messages_low = [[] for _ in range(n_nodes)]
#     messages_high = [[] for _ in range(n_nodes)]# list of lists of messages
#     for idx in range(n_edges):
#         certain_edge = True if edge_status[idx] == 1 else False
#         sidx = index_sources[idx]
#         tidx = index_targets[idx]
#         if certain_edge is True:
#             msg_low, msg_high = lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :] * elow[idx], lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :] * ehigh[idx]
#             # msg_low, msg_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#             #                                                 (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#             #                                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#         else:
#             raise NotImplementedError
#             msg_low, msg_high = abs_bwd_phi_product_with_bot((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#                                             (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#                                             (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # m_low, m_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#             #                                 (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#             #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # bot_low, bot_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#             #                                 (None, None, xlow[sidx], xhigh[sidx]),
#             #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#             # msg_low, msg_high = least_upper_bound(m_low, m_high, bot_low, bot_high)
#         messages_low[index_targets[idx]].append(msg_low)
#         messages_high[index_targets[idx]].append(msg_high)
#
#     embeddings_low = torch.concat(
#         [abs_bwd_sigma_sum(m, lA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_low)])
#
#     embeddings_high = torch.concat(
#         [abs_bwd_sigma_sum(m, uA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_high)])
#
#     lAx = embeddings_low
#     uAx = embeddings_high
#
#     # class NodeValues:
#     #     def __init__(self):
#     #         self.current_nonzero_idx = torch.empty((0,))
#     #         self.indices = {}
#     #         self.lows = {}
#     #         self.highs = {}
#     #
#     #     def get_node_values(self, nonzero_idx):
#     #         if not torch.equal(self.current_nonzero_idx, nonzero_idx):
#     #             self.current_nonzero_idx = nonzero_idx
#     #             self.indices, self.lows, self.highs = {}, {}, {}
#     #             for nzi in nonzero_idx:
#     #                 nzi = nzi.item()
#     #                 _node = nzi // n_node_features_pre
#     #                 # i = nzi % n_node_features_pre
#     #                 edge_idx = np.where(index_targets == _node)[0]
#     #                 self.indices[_node] = []
#     #                 self.lows[_node] = []
#     #                 self.highs[_node] = []
#     #                 for idx in edge_idx:
#     #                     sidx = index_sources[idx]
#     #                     # newmlow, newmhigh = abs_phi_prod2((elow[idx], ehigh[idx], xlow[sidx][i], xhigh[sidx][i]))
#     #                     self.indices[_node].append(sidx)
#     #                     self.lows[_node].append(elow[idx])
#     #                     self.highs[_node].append(ehigh[idx])
#     #
#     #         for j in [nnz.item() for nnz in nonzero_idx]:
#     #             node = j // n_node_features_pre
#     #             values_low = torch.zeros((1, n_vars_pre))
#     #             values_high = torch.zeros((1, n_vars_pre))
#     #             for _idx, low, high in zip(self.indices[node], self.lows[node], self.highs[node]):
#     #                 values_low[0][_idx * n_node_features_pre + j % n_node_features_pre] = low
#     #                 values_high[0][_idx * n_node_features_pre + j % n_node_features_pre] = high
#     #             yield values_low, values_high
#
#     reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
#     reshaped_lax = lAx.view((n_vars_pre, 1, -1))
#     reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
#     reshaped_uax = uAx.view((n_vars_pre, 1, -1))
#     # node_val_obj = NodeValues()
#
#     lA, uA = [], []
#     for node in range(n_nodes):
#         # node_val = get_node_values(node)
#         for v_offset in range(n_node_features_post):
#             v = node * n_node_features_post + v_offset
#             print(v, 'out of', n_vars_post)
#             non_zero_indices = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#             if non_zero_indices.nelement() != 0:
#                 prev_values_la = reshaped_last_la[v][0][non_zero_indices]
#                 prev_values_ua = reshaped_last_ua[v][0][non_zero_indices]
#                 # node_val = node_val_obj.get_node_values(non_zero_indices)
#                 # values_la, values_ua = list(zip(*get_node_values(non_zero_indices)))
#                 values_la, values_ua = torch.unbind(reshaped_lax[non_zero_indices]), torch.unbind(reshaped_uax[non_zero_indices])
#                 # assert np.array_equal(values_la_1, values_la)
#                 rec_value_la = reshaped_last_la[v]
#                 rec_value_ua = reshaped_last_ua[v]
#                 messages = []
#                 for coeff_la, coeff_ua, mat_la, mat_ua in zip(prev_values_la, prev_values_ua, values_la, values_ua):
#                     msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                     messages.append((msg_low, msg_high))
#                 embeddings_low, embeddings_high = abs_sigma(messages, (rec_value_la, rec_value_ua))
#             else:
#                 embeddings_low = torch.zeros((1, n_vars_pre))
#                 embeddings_high = torch.zeros((1, n_vars_pre))
#             lA.append(embeddings_low)
#             uA.append(embeddings_high)
#
#     lAx = torch.vstack(lA).view((len(lA), 1, n_nodes, n_node_features_pre))
#     uAx = torch.vstack(uA).view((len(uA), 1, n_nodes, n_node_features_pre))
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.

# @torch.no_grad()
# def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     # abs_bwd_phi = function_store_phi.poly_transformer_bwd
#     # abs_bwd_sigma = function_store_sigma.poly_transformer_bwd
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
#     n_node_features_pre = last_lA.shape[-1]
#     n_node_features_post = n_vars_post // n_nodes
#     n_edges = e.output_shape[0]
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     edge_status = a.forward_value[0][-1]
#     elow, ehigh = e.interval[0], e.interval[1]
#     xlow, xhigh = x.interval[0][0], x.interval[1][0]
#
#
#     lA = torch.eye(n_vars_pre).view((n_vars_pre, 1, n_nodes, n_node_features_pre))
#     uA = lA
#
#     messages_low = [[] for _ in range(n_nodes)]
#     messages_high = [[] for _ in range(n_nodes)]# list of lists of messages
#     for idx in range(n_edges):
#         certain_edge = True if edge_status[idx] == 1 else False
#         sidx = index_sources[idx]
#         tidx = index_targets[idx]
#         msg_low, msg_high = lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :] * elow[idx], lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :] * ehigh[idx]
#         # msg_low, msg_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#         #                                                 (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#         #                                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#
#         messages_low[index_targets[idx]].append(msg_low)
#         messages_high[index_targets[idx]].append(msg_high)
#
#     embeddings_low = torch.concat(
#         [abs_bwd_sigma_sum(m, lA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_low)])
#
#     embeddings_high = torch.concat(
#         [abs_bwd_sigma_sum(m, uA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_high)])
#
#     lAx = embeddings_low
#     uAx = embeddings_high
#
#     # class NodeValues:
#     #     def __init__(self):
#     #         self.current_nonzero_idx = torch.empty((0,))
#     #         self.indices = {}
#     #         self.lows = {}
#     #         self.highs = {}
#     #
#     #     def get_node_values(self, nonzero_idx):
#     #         if not torch.equal(self.current_nonzero_idx, nonzero_idx):
#     #             self.current_nonzero_idx = nonzero_idx
#     #             self.indices, self.lows, self.highs = {}, {}, {}
#     #             for nzi in nonzero_idx:
#     #                 nzi = nzi.item()
#     #                 _node = nzi // n_node_features_pre
#     #                 # i = nzi % n_node_features_pre
#     #                 edge_idx = np.where(index_targets == _node)[0]
#     #                 self.indices[_node] = []
#     #                 self.lows[_node] = []
#     #                 self.highs[_node] = []
#     #                 for idx in edge_idx:
#     #                     sidx = index_sources[idx]
#     #                     # newmlow, newmhigh = abs_phi_prod2((elow[idx], ehigh[idx], xlow[sidx][i], xhigh[sidx][i]))
#     #                     self.indices[_node].append(sidx)
#     #                     self.lows[_node].append(elow[idx])
#     #                     self.highs[_node].append(ehigh[idx])
#     #
#     #         for j in [nnz.item() for nnz in nonzero_idx]:
#     #             node = j // n_node_features_pre
#     #             values_low = torch.zeros((1, n_vars_pre))
#     #             values_high = torch.zeros((1, n_vars_pre))
#     #             for _idx, low, high in zip(self.indices[node], self.lows[node], self.highs[node]):
#     #                 values_low[0][_idx * n_node_features_pre + j % n_node_features_pre] = low
#     #                 values_high[0][_idx * n_node_features_pre + j % n_node_features_pre] = high
#     #             yield values_low, values_high
#
#     reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
#     reshaped_lax = lAx.view((n_vars_pre, 1, -1))
#     reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
#     reshaped_uax = uAx.view((n_vars_pre, 1, -1))
#     # node_val_obj = NodeValues()
#
#     lA, uA = [], []
#     for node in range(n_nodes):
#         # node_val = get_node_values(node)
#         for v_offset in range(n_node_features_post):
#             v = node * n_node_features_post + v_offset
#             print(v, 'out of', n_vars_post)
#             non_zero_indices = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#             if non_zero_indices.nelement() != 0:
#                 prev_values_la = reshaped_last_la[v][0][non_zero_indices]
#                 prev_values_ua = reshaped_last_ua[v][0][non_zero_indices]
#                 # node_val = node_val_obj.get_node_values(non_zero_indices)
#                 # values_la, values_ua = list(zip(*get_node_values(non_zero_indices)))
#                 values_la, values_ua = torch.unbind(reshaped_lax[non_zero_indices]), torch.unbind(reshaped_uax[non_zero_indices])
#                 # assert np.array_equal(values_la_1, values_la)
#                 rec_value_la = reshaped_last_la[v]
#                 rec_value_ua = reshaped_last_ua[v]
#                 messages = []
#                 for coeff_la, coeff_ua, mat_la, mat_ua in zip(prev_values_la, prev_values_ua, values_la, values_ua):
#                     msg_low, msg_high = abs_phi_product((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                     messages.append((msg_low, msg_high))
#                 embeddings_low, embeddings_high = abs_sigma(messages, (rec_value_la, rec_value_ua))
#             else:
#                 embeddings_low = torch.zeros((1, n_vars_pre))
#                 embeddings_high = torch.zeros((1, n_vars_pre))
#             lA.append(embeddings_low)
#             uA.append(embeddings_high)
#
#     lAx = torch.vstack(lA).view((len(lA), 1, n_nodes, n_node_features_pre))
#     uAx = torch.vstack(uA).view((len(uA), 1, n_nodes, n_node_features_pre))
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.

def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, last_lA, last_uA, x, a, e):
    if use_optimized_gcn:
        return bwd_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e)
    else:
        return bwd_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e)


# @torch.no_grad()
# def bwd_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
#     n_node_features_pre = n_vars_pre // n_nodes
#     n_node_features_post = n_vars_post // n_nodes
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     edge_status = a.forward_value[0][-1]
#     elow, ehigh = e.interval[0], e.interval[1]
#
#     class NodeValues:
#         def __init__(self):
#             self.current_nonzero_idx = torch.empty((0,))
#             self.indices = {}
#             self.lows = {}
#             self.highs = {}
#
#         def get_node_values(self, nonzero_idx):
#             if not torch.equal(self.current_nonzero_idx, nonzero_idx):
#                 self.current_nonzero_idx = nonzero_idx
#                 self.indices, self.lows, self.highs = {}, {}, {}
#                 for nzi in nonzero_idx:
#                     nzi = nzi.item()
#                     _node = nzi // n_node_features_pre
#                     # i = nzi % n_node_features_pre
#                     edge_idx = np.where(index_targets == _node)[0]
#                     self.indices[_node] = []
#                     self.lows[_node] = []
#                     self.highs[_node] = []
#                     for idx in edge_idx:
#                         certain_edge = True if edge_status[idx] == 1 else False
#                         sidx = index_sources[idx]
#                         self.indices[_node].append(sidx)
#                         if certain_edge is True:
#                             self.lows[_node].append(elow[idx])
#                             self.highs[_node].append(ehigh[idx])
#                         else:
#                             bot_low, bot_high = function_store_sigma.identity_like(elow[idx]) # abs_phi((elow[idx], ehigh[idx]), None, (elow[idx], ehigh[idx]))
#                             self.lows[_node].append(torch.min(elow[idx], bot_low))
#                             self.highs[_node].append(torch.max(ehigh[idx], bot_high))
#
#             nonzero_idx = [nnz.item() for nnz in nonzero_idx]
#             values_low = torch.zeros((1, n_vars_pre))
#             values_high = torch.zeros((1, n_vars_pre))
#             for j in nonzero_idx:
#                 values_low.zero_()
#                 values_high.zero_()
#                 node = j // n_node_features_pre
#                 offset = j % n_node_features_pre
#                 idx_array = torch.tensor(self.indices[node])
#                 pos_array = idx_array * n_node_features_pre + offset
#                 values_low[0][pos_array] = torch.tensor(self.lows[node])
#                 values_high[0][pos_array] = torch.tensor(self.highs[node])
#                 yield values_low, values_high
#
#     reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
#     reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
#     node_val_obj = NodeValues()
#     id_value_low, id_value_high = function_store_sigma.identity_like(reshaped_last_la[0])
#
#     lA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
#     uA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
#     for node in tqdm(range(n_nodes)):
#         for v_offset in range(n_node_features_post):
#             v = node * n_node_features_post + v_offset
#             non_zero_indices_l = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#             non_zero_indices_u = torch.nonzero(reshaped_last_ua[v], as_tuple=True)[1]
#             if non_zero_indices_u.nelement() != 0 and non_zero_indices_l.nelement() != 0:
#                 prev_values_la = reshaped_last_la[v][0][non_zero_indices_u]
#                 prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
#                 node_val = node_val_obj.get_node_values(non_zero_indices_u)
#                 rec_value_la = reshaped_last_la[v]
#                 rec_value_ua = reshaped_last_ua[v]
#                 emb_low, emb_high = id_value_low, id_value_high
#                 for coeff_la, coeff_ua, (mat_la, mat_ua) in zip(prev_values_la, prev_values_ua, node_val):
#                     msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                     emb_low, emb_high = abs_sigma([(emb_low, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
#                 emb_low, emb_high = abs_sigma([(emb_low, emb_high), (id_value_low, id_value_high)], (rec_value_la, rec_value_ua))
#                 embeddings_low, embeddings_high = emb_low, emb_high
#             elif non_zero_indices_l.nelement() != 0:
#                 embeddings_high = torch.zeros((1, n_vars_pre))
#                 prev_values_la = reshaped_last_ua[v][0][non_zero_indices_l]
#                 node_val = node_val_obj.get_node_values(non_zero_indices_l)
#                 rec_value_la = reshaped_last_la[v]
#                 emb_low = id_value_low
#                 for coeff_la, (mat_la, mat_ua) in zip(prev_values_la, node_val):
#                     msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_la), (rec_value_la, rec_value_la))
#                     emb_low, emb_high = abs_sigma([(emb_low, emb_low), (msg_low, msg_high)], (id_value_low, id_value_high))
#                 emb_low, emb_high = abs_sigma([(emb_low, emb_low), (id_value_low, id_value_high)], (rec_value_la, rec_value_la))
#                 embeddings_low = emb_low
#             elif non_zero_indices_u.nelement() != 0:
#                 embeddings_low = torch.zeros((1, n_vars_pre))
#                 prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
#                 node_val = node_val_obj.get_node_values(non_zero_indices_u)
#                 rec_value_ua = reshaped_last_ua[v]
#                 emb_high = id_value_high
#                 for coeff_ua, (mat_la, mat_ua) in zip(prev_values_ua, node_val):
#                     msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_ua, coeff_ua), (rec_value_ua, rec_value_ua))
#                     emb_low, emb_high = abs_sigma([(emb_high, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
#                 emb_low, emb_high = abs_sigma([(emb_high, emb_high), (id_value_low, id_value_high)], (rec_value_ua, rec_value_ua))
#                 embeddings_high = emb_high
#             else:
#                 embeddings_low = torch.zeros((1, n_vars_pre))
#                 embeddings_high = torch.zeros((1, n_vars_pre))
#             lA[v] = embeddings_low
#             uA[v] = embeddings_high
#
#     lAx = lA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))
#     uAx = uA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))
#
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.



@torch.no_grad()
def bwd_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
    """ Backward mode bound propagation """
    abs_phi = function_store_phi.interval_transformer
    abs_sigma = function_store_sigma.interval_transformer
    n_nodes = x.output_shape[1]
    n_vars_post = last_lA.shape[0]
    n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
    n_node_features_pre = n_vars_pre // n_nodes
    n_node_features_post = n_vars_post // n_nodes
    index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
    index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
    edge_status = a.forward_value[0][-1]
    elow, ehigh = e.interval[0], e.interval[1]
    xlow, xhigh = x.interval[0], x.interval[1]

    class NodeValues:
        def __init__(self, xl, xh):
            self.xlow = xl
            self.xhigh = xh
            self.current_nonzero_idx = torch.empty((0,))
            self.indices = {}
            self.lows = {}
            self.highs = {}

        def get_node_values(self, nonzero_idx):
            if not torch.equal(self.current_nonzero_idx, nonzero_idx):
                self.current_nonzero_idx = nonzero_idx
                self.indices, self.lows, self.highs = {}, {}, {}
                for nzi in nonzero_idx:
                    nzi = nzi.item()
                    _node = nzi // n_node_features_pre
                    offset = nzi % n_node_features_pre
                    edge_idx = np.where(index_targets == _node)[0]
                    self.indices[(_node, offset)] = []
                    self.lows[(_node, offset)] = []
                    self.highs[(_node, offset)] = []
                    for idx in edge_idx:
                        certain_edge = True if edge_status[idx] == 1 else False
                        sidx = index_sources[idx]
                        self.indices[(_node, offset)].append(sidx)
                        if certain_edge is True:
                            e0, e1 = elow[idx], ehigh[idx]
                            src_node = index_sources[idx]
                            xl, xh = self.xlow[0][src_node][offset], self.xhigh[0][src_node][offset]
                            argmin = np.argmin([e0 * xl, e0 * xh, e1 * xl,  e1 * xh])
                            emin, emax = (e0, e1) if argmin <= 1 else (e1, e0)
                            self.lows[(_node, offset)].append(emin)
                            self.highs[(_node, offset)].append(emax)
                        else:
                            e0, e1 = elow[idx], ehigh[idx]
                            src_node = index_sources[idx]
                            xl, xh = self.xlow[0][src_node][offset], self.xhigh[0][src_node][offset]
                            argmin = np.argmin([e0 * xl, e0 * xh, e1 * xl,  e1 * xh])
                            emin, emax = (e0, e1) if argmin <= 1 else (e1, e0)
                            bot_low, bot_high = function_store_sigma.identity_like(elow[idx]) # abs_phi((elow[idx], ehigh[idx]), None, (elow[idx], ehigh[idx]))
                            self.lows[(_node, offset)].append(torch.min(emin, bot_low))
                            self.highs[(_node, offset)].append(torch.max(emax, bot_high))

            nonzero_idx = [nnz.item() for nnz in nonzero_idx]
            values_low = torch.zeros((1, n_vars_pre))
            values_high = torch.zeros((1, n_vars_pre))
            for j in nonzero_idx:
                values_low.zero_()
                values_high.zero_()
                node = j // n_node_features_pre
                offset = j % n_node_features_pre
                idx_array = torch.tensor(self.indices[(node, offset)])
                pos_array = idx_array * n_node_features_pre + offset
                values_low[0][pos_array] = torch.tensor(self.lows[(node, offset)])
                values_high[0][pos_array] = torch.tensor(self.highs[(node, offset)])
                yield values_low, values_high

    reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
    reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
    node_val_obj = NodeValues(xlow, xhigh)
    id_value_low, id_value_high = function_store_sigma.identity_like(reshaped_last_la[0])

    lA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
    uA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
    for node in tqdm(range(n_nodes)):
        for v_offset in range(n_node_features_post):
            v = node * n_node_features_post + v_offset
            non_zero_indices_l = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
            non_zero_indices_u = torch.nonzero(reshaped_last_ua[v], as_tuple=True)[1]
            if non_zero_indices_u.nelement() != 0 and non_zero_indices_l.nelement() != 0:
                prev_values_la = reshaped_last_la[v][0][non_zero_indices_u]
                prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
                node_val = node_val_obj.get_node_values(non_zero_indices_u)
                rec_value_la = reshaped_last_la[v]
                rec_value_ua = reshaped_last_ua[v]
                emb_low, emb_high = id_value_low, id_value_high
                for coeff_la, coeff_ua, (mat_la, mat_ua) in zip(prev_values_la, prev_values_ua, node_val):
                    if coeff_la >=0:
                        msg_low = mat_la * coeff_la
                    else:
                        msg_low = mat_ua * coeff_la
                    if coeff_ua >= 0:
                        msg_high = mat_ua * coeff_ua
                    else:
                        msg_high = mat_la * coeff_ua
                    emb_low, emb_high = abs_sigma([(emb_low, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_low, emb_high), (id_value_low, id_value_high)], (rec_value_la, rec_value_ua))
                embeddings_low, embeddings_high = emb_low, emb_high
            elif non_zero_indices_l.nelement() != 0:
                embeddings_high = torch.zeros((1, n_vars_pre))
                prev_values_la = reshaped_last_ua[v][0][non_zero_indices_l]
                node_val = node_val_obj.get_node_values(non_zero_indices_l)
                rec_value_la = reshaped_last_la[v]
                emb_low = id_value_low
                for coeff_la, (mat_la, mat_ua) in zip(prev_values_la, node_val):
                    if coeff_la >= 0:
                        msg_low, msg_high = mat_la * coeff_la, mat_ua * coeff_la # abs_phi((mat_la, mat_ua), (coeff_la, coeff_la), (rec_value_la, rec_value_la))
                    else:
                        msg_low, msg_high = mat_ua * coeff_la, mat_la * coeff_la
                    emb_low, emb_high = abs_sigma([(emb_low, emb_low), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_low, emb_low), (id_value_low, id_value_high)], (rec_value_la, rec_value_la))
                embeddings_low = emb_low
            elif non_zero_indices_u.nelement() != 0:
                embeddings_low = torch.zeros((1, n_vars_pre))
                prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
                node_val = node_val_obj.get_node_values(non_zero_indices_u)
                rec_value_ua = reshaped_last_ua[v]
                emb_high = id_value_high
                for coeff_ua, (mat_la, mat_ua) in zip(prev_values_ua, node_val):
                    if coeff_ua >= 0:
                        msg_low, msg_high = mat_la * coeff_ua, mat_ua * coeff_ua # abs_phi((mat_la, mat_ua), (coeff_ua, coeff_ua), (rec_value_ua, rec_value_ua))
                    else:
                        msg_low, msg_high = mat_ua * coeff_ua, mat_la * coeff_ua
                    emb_low, emb_high = abs_sigma([(emb_high, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_high, emb_high), (id_value_low, id_value_high)], (rec_value_ua, rec_value_ua))
                embeddings_high = emb_high
            else:
                embeddings_low = torch.zeros((1, n_vars_pre))
                embeddings_high = torch.zeros((1, n_vars_pre))
            lA[v] = embeddings_low
            uA[v] = embeddings_high

    lAx = lA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))
    uAx = uA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))

    return [(lAx, uAx), (None, None), (None, None)], 0., 0.

@torch.no_grad()
def bwd_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
    """ Backward mode bound propagation """
    if src_idx == 1:
        e.lower = e.lower.transpose(-1, -2)
        e.upper = e.upper.transpose(-1, -2)

    input_lb = [e.lower, x.lower]
    input_ub = [e.upper, x.upper]

    input_lb[0] = input_lb[0].transpose(-2, -1)
    input_ub[0] = input_ub[0].transpose(-2, -1)
    input_lb[1] = input_lb[1].transpose(-2, -1)
    input_ub[1] = input_ub[1].transpose(-2, -1)

    input_lb[0] = input_lb[0].unsqueeze(-2)
    input_ub[0] = input_ub[0].unsqueeze(-2)
    input_lb[1] = input_lb[1].unsqueeze(-3)
    input_ub[1] = input_ub[1].unsqueeze(-3)

    # (alpha_l, beta_l, gamma_l,
    #  alpha_u, beta_u, gamma_u) = MulHelper.interpolated_relaxation(input_lb[0], input_ub[0], input_lb[1], input_ub[1])

    x_l, x_u, y_l, y_u = input_lb[0], input_ub[0], input_lb[1], input_ub[1]

    alpha_l, beta_l, gamma_l = y_l, x_l, -y_l * x_l
    alpha_u, beta_u, gamma_u = y_u, x_l, -y_u * x_l

    # x_shape = input_lb[0].size()
    gamma_l = torch.sum(gamma_l, dim=-1)
    gamma_u = torch.sum(gamma_u, dim=-1)

    # if len(e.output_shape) != 2 and len(e.output_shape) == len(x.output_shape):
    # dim_y = [-3]
    # elif len(x.output_shape) == 2:
    #     dim_y = list(range(2, 2 + len(x_shape) - 2))
    # else:
    #     raise NotImplementedError


    @torch.jit.script
    def propagate_A_xy(last_A, alpha_pos, alpha_neg,
                       beta_pos, beta_neg):
        # last_uA has size (batch, spec, output)
        last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
        last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
        # alpha_u has size (batch, spec, output, input)
        # uA_x has size (batch, spec, input).
        A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) +
                alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
        # beta_u has size (batch, spec, output, input)
        # uA_y is for weight matrix, with parameter size (output, input)
        # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
        # TODO (for zhouxing/qirui): generalize multiply_by_A_signs() to calculate A_x,
        # so last_A_pos and last_A_neg are not needed. This saves memory.
        d_pos = beta_pos.contiguous()
        d_neg = beta_neg.contiguous()
        A = last_A.unsqueeze(-1)

        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)

        # Initialize output tensor
        A_new = torch.zeros((A_pos.shape[0], A_pos.shape[1], A_pos.shape[3], A_pos.shape[2]), device=A_pos.device)

        # Loop over i (the summation axis)
        for i in range(A_pos.shape[2]):
            # Extract weights for this i across all j
            w_pos = d_pos[0, i, 0, :]  # shape: [80]
            w_neg = d_neg[0, i, 0, :]  # shape: [80]

            # Extract A_pos and A_neg slice at i
            A_pos_i = A_pos[:, 0, i, :, 0]  # shape: [560, 32]
            A_neg_i = A_neg[:, 0, i, :, 0]  # shape: [560, 32]

            # Reshape for broadcasting: [560, 1, 32, 1]
            A_pos_i = A_pos_i.unsqueeze(1).unsqueeze(-1)
            A_neg_i = A_neg_i.unsqueeze(1).unsqueeze(-1)

            # Reshape weights: [1, 1, 1, 80]
            w_pos = w_pos.view(1, 1, 1, A_pos.shape[2])
            w_neg = w_neg.view(1, 1, 1, A_pos.shape[2])

            # Accumulate weighted contribution
            A_new += A_pos_i * w_pos + A_neg_i * w_neg

        A_y = A_new
        # if len(dim_y) != 0:
        #     A_y = torch.sum(A_y, dim=dim_y)
        return A_x, A_y

    @torch.jit.script
    def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
        if last_A is None:
            return None, None, 0

        A_x, A_y = propagate_A_xy(
            last_A, alpha_pos, alpha_neg, beta_pos, beta_neg)

        # last_uA has size (batch, spec, output)
        # gamma_u has size (batch, output, 1)
        # ubias has size (batch, spec, 1)
        bias = ( torch.einsum('sb...,b...->sb', last_A.clamp(min=0), gamma_pos)
                + torch.einsum('sb...,b...->sb', last_A.clamp(max=0), gamma_neg)
        )
        return A_x, A_y, bias

    lA_x, lA_y, lbias = _bound_oneside(
        last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
    uA_x, uA_y, ubias = _bound_oneside(
        last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

    results = [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    lA_y = results[0][1][0].transpose(-1, -2)
    uA_y = results[0][1][1].transpose(-1, -2)
    lA_e = results[0][0][0].transpose(-1, -2)
    uA_e = results[0][0][1].transpose(-1, -2)

    if isinstance(results[1], tuple):
        lbias = (results[1][0], results[1][1].transpose(-1, -2))
    else:
        lbias = results[1]
    if isinstance(results[2], tuple):
        ubias = (results[2][0], results[2][1].transpose(-1, -2))
    else:
        ubias = results[2]

    # for edge abs only
    # edge_status = a.forward_value[0]
    # uncertain_edges = edge_status < 0
    #
    # pos_mask = uncertain_edges & (lA_e > 0)
    # pos_indices = torch.nonzero(pos_mask, as_tuple=False)
    # pos_i = pos_indices[:, 2]
    # pos_j = pos_indices[:, 3]
    # neg_mask = uncertain_edges & (lA_e < 0)
    # neg_indices = torch.nonzero(neg_mask, as_tuple=False)
    # neg_i = neg_indices[:, 2]
    # neg_j = neg_indices[:, 3]
    #
    # weight_pos = (torch.minimum(e.lower[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_pos = weight_pos[pos_i, pos_j]
    # weight_neg = (torch.maximum(e.lower[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_neg = weight_neg[neg_i, neg_j]
    #
    # lA_e[pos_mask] *= selected_weight_pos
    # lA_e[neg_mask] *= selected_weight_neg
    #
    #
    # pos_mask = uncertain_edges & (uA_e > 0)
    # pos_indices = torch.nonzero(pos_mask, as_tuple=False)
    # pos_i = pos_indices[:, 2]
    # pos_j = pos_indices[:, 3]
    # neg_mask = uncertain_edges & (uA_e < 0)
    # neg_indices = torch.nonzero(neg_mask, as_tuple=False)
    # neg_i = neg_indices[:, 2]
    # neg_j = neg_indices[:, 3]
    #
    # weight_pos = (torch.maximum(e.upper[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_pos = weight_pos[pos_i, pos_j]
    # weight_neg = (torch.minimum(e.upper[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_neg = weight_neg[neg_i, neg_j]
    #
    # uA_e[pos_mask] *= selected_weight_pos
    # uA_e[neg_mask] *= selected_weight_neg


    if src_idx == 1:
        lA_e = lA_e.transpose(-1, -2)
        uA_e = uA_e.transpose(-1, -2)

    return [(lA_y, uA_y), (None, None), (lA_e, uA_e)], lbias, ubias


# @torch.no_grad()
# def bwd_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     input_lb = [xi.lower for xi in [e, x]]
#     input_ub = [xi.upper for xi in [e, x]]
#
#     input_lb[0] = input_lb[0].transpose(-2, -1)
#     input_ub[0] = input_ub[0].transpose(-2, -1)
#     input_lb[1] = input_lb[1].transpose(-2, -1)
#     input_ub[1] = input_ub[1].transpose(-2, -1)
#
#     input_lb[0] = input_lb[0].unsqueeze(-2)
#     input_ub[0] = input_ub[0].unsqueeze(-2)
#     input_lb[1] = input_lb[1].unsqueeze(-3)
#     input_ub[1] = input_ub[1].unsqueeze(-3)
#
#     # (alpha_l, beta_l, gamma_l,
#     #  alpha_u, beta_u, gamma_u) = MulHelper.interpolated_relaxation(input_lb[0], input_ub[0], input_lb[1], input_ub[1])
#
#     x_l, x_u, y_l, y_u = input_lb[0], input_ub[0], input_lb[1], input_ub[1]
#
#     alpha_l, beta_l, gamma_l = y_l, x_l, -y_l * x_l
#     alpha_u, beta_u, gamma_u = y_u, x_u, -y_u * x_l
#
#     x_shape = input_lb[0].size()
#     gamma_l = torch.sum(gamma_l, dim=-1)
#     gamma_u = torch.sum(gamma_u, dim=-1)
#
#     if len(e.output_shape) != 2 and len(e.output_shape) == len(x.output_shape):
#         dim_y = [-3]
#     elif len(x.output_shape) == 2:
#         dim_y = list(range(2, 2 + len(x_shape) - 2))
#     else:
#         raise NotImplementedError
#
#
#     @torch.no_grad()
#     #@torch.jit.script
#     def clamp_mutiply_forward(A: Tensor, d_pos: Tensor, d_neg: Tensor,
#                               b_pos: Optional[Tensor], b_neg: Optional[Tensor], patches_mode: bool,
#                               reduce_bias: bool = False, same_slope: bool = False
#                               ) -> Tuple[Tensor, Tensor]:
#         """Forward operations; actually the same as the reference implementation."""
#         A_pos = A.clamp(min=0)
#         A_neg = A.clamp(max=0)
#         if same_slope:
#             # "same-slope" option is enabled; lower and upper bounds use the same A.
#             A_new = d_pos * A
#         else:
#             # print("d_pos shape:", d_pos.shape)
#             # print("A_pos shape:", A_pos.shape)
#             # print("d_neg shape:", d_neg.shape)
#             # print("A_neg shape:", A_neg.shape)
#
#             # Initialize output tensor
#             A_new = torch.zeros((A_pos.shape[0], A_pos.shape[1], A_pos.shape[3], A_pos.shape[2]), device=A_pos.device)
#
#             # Loop over i (the summation axis)
#             for i in range(A_pos.shape[2]):
#                 # Extract weights for this i across all j
#                 w_pos = d_pos[0, i, 0, :]  # shape: [80]
#                 w_neg = d_neg[0, i, 0, :]  # shape: [80]
#
#                 # Extract A_pos and A_neg slice at i
#                 A_pos_i = A_pos[:, 0, i, :, 0]  # shape: [560, 32]
#                 A_neg_i = A_neg[:, 0, i, :, 0]  # shape: [560, 32]
#
#                 # Reshape for broadcasting: [560, 1, 32, 1]
#                 A_pos_i = A_pos_i.unsqueeze(1).unsqueeze(-1)
#                 A_neg_i = A_neg_i.unsqueeze(1).unsqueeze(-1)
#
#                 # Reshape weights: [1, 1, 1, 80]
#                 w_pos = w_pos.view(1, 1, 1, A_pos.shape[2])
#                 w_neg = w_neg.view(1, 1, 1, A_pos.shape[2])
#
#                 # Accumulate weighted contribution
#                 A_new += A_pos_i * w_pos + A_neg_i * w_neg
#
#             # print("A_new shape:", A_new.shape)
#             # A_new = d_pos * A_pos + d_neg * A_neg
#             # print("A_new_b shape:", A_new_b.shape)
#             # A_new = torch.sum(A_new, dim=[-3])
#             # print("A_new_b shape:", A_new_b.shape)
#
#
#         # bias_pos = bias_neg = torch.zeros(
#         #     (), dtype=A_new.dtype, device=A_new.device)
#         # if b_pos is not None:
#         #     if not reduce_bias:
#         #         bias_pos = A_pos * b_pos
#         #     else:
#         #         if patches_mode:
#         #             bias_pos = torch.einsum('sb...chw,sb...chw->sb...', A_pos, b_pos)
#         #         else:
#         #             bias_pos = torch.einsum('sb...,sb...->sb', A_pos, b_pos)
#         # if b_neg is not None:
#         #     if not reduce_bias:
#         #         bias_neg = A_neg * b_neg
#         #     else:
#         #         if patches_mode:
#         #             bias_neg = torch.einsum('sb...chw,sb...chw->sb...', A_neg, b_neg)
#         #         else:
#         #             bias_neg = torch.einsum('sb...,sb...->sb', A_neg, b_neg)
#         return A_new, 0.
#
#     def multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg, contiguous='auto',
#                             reduce_bias=True, same_slope=False):
#         # For dense mode, convert d_pos and d_neg to contiguous tensor by default.
#         d_pos = d_pos.contiguous()
#         d_neg = d_neg.contiguous()
#         return clamp_mutiply_forward(
#             A, d_pos, d_neg, b_pos, b_neg, False, reduce_bias, same_slope)
#
#
#     # @torch.jit.script
#     def propagate_A_xy(last_A, alpha_pos, alpha_neg,
#                        beta_pos, beta_neg,
#                        dim_y):
#         # last_uA has size (batch, spec, output)
#         last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
#         last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
#         # alpha_u has size (batch, spec, output, input)
#         # uA_x has size (batch, spec, input).
#         A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) +
#                 alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
#         # beta_u has size (batch, spec, output, input)
#         # uA_y is for weight matrix, with parameter size (output, input)
#         # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
#         # TODO (for zhouxing/qirui): generalize multiply_by_A_signs() to calculate A_x,
#         # so last_A_pos and last_A_neg are not needed. This saves memory.
#
#         A_y, _ = multiply_by_A_signs(last_A.unsqueeze(-1), beta_pos, beta_neg, alpha_pos.transpose(-1, -2), alpha_neg.transpose(-1, -2))
#         # if len(dim_y) != 0:
#         #     A_y = torch.sum(A_y, dim=dim_y)
#         return A_x, A_y
#
#     #
#     def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
#         if last_A is None:
#             return None, None, 0
#
#         A_x, A_y = propagate_A_xy(
#             last_A, alpha_pos, alpha_neg, beta_pos, beta_neg, dim_y)
#
#         # last_uA has size (batch, spec, output)
#         # gamma_u has size (batch, output, 1)
#         # ubias has size (batch, spec, 1)
#         bias = ( torch.einsum('sb...,b...->sb', last_A.clamp(min=0), gamma_pos)
#                 + torch.einsum('sb...,b...->sb', last_A.clamp(max=0), gamma_neg)
#         )
#         return A_x, A_y, bias
#         # return None, None, bias
#
#     lA_x, lA_y, lbias = _bound_oneside(
#         last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
#     uA_x, uA_y, ubias = _bound_oneside(
#         last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)
#     #
#     results = [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias
#
#     lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
#     uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
#     # lA_e = results[0][0][0].transpose(-1, -2)
#     # uA_e = results[0][0][1].transpose(-1, -2)
#
#     # if isinstance(results[1], tuple):
#     #     lbias = (results[1][0], results[1][1].transpose(-1, -2))
#     # else:
#     #     lbias = results[1]
#     # if isinstance(results[2], tuple):
#     #     ubias = (results[2][0], results[2][1].transpose(-1, -2))
#     # else:
#     #     ubias = results[2]
#
#     lbias = 0.
#     ubias = 0.
#     lA_e = None
#     uA_e = None
#
#
#     return [(lA_y, uA_y), (None, None), (lA_e, uA_e)], lbias, ubias


# @torch.no_grad()
# @profile
# def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
#     """ Backward mode bound propagation """
#     abs_bwd_phi = function_store_phi.poly_transformer_bwd
#     abs_bwd_sigma = function_store_sigma.poly_transformer_bwd
#     abs_phi = function_store_phi.interval_transformer
#     abs_sigma = function_store_sigma.interval_transformer
#     n_nodes = x.output_shape[1]
#     n_vars_post = last_lA.shape[0]
#     n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
#     n_node_features_pre = n_vars_pre // n_nodes # last_lA.shape[-1]
#     n_node_features_post = n_vars_post // n_nodes
#     n_edges = e.output_shape[0]
#     index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
#     index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
#     edge_status = a.forward_value[0][-1]
#     elow, ehigh = e.interval[0], e.interval[1]
#     xlow, xhigh = x.interval[0][0], x.interval[1][0]
#
#
#     # lA = torch.eye(n_vars_pre).view((n_vars_pre, 1, n_nodes, n_node_features_pre))
#     # uA = lA
#     #
#     # messages_low = [[] for _ in range(n_nodes)]
#     # messages_high = [[] for _ in range(n_nodes)]# list of lists of messages
#     # for idx in range(n_edges):
#     #     certain_edge = True if edge_status[idx] == 1 else False
#     #     sidx = index_sources[idx]
#     #     tidx = index_targets[idx]
#     #     if certain_edge is True:
#     #         msg_low, msg_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#     #                                                         (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#     #                                                         (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#     #     else:
#     #         msg_low, msg_high = abs_bwd_phi_product_with_bot((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#     #                                         (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#     #                                         (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#     #         # m_low, m_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#     #         #                                 (elow[idx], ehigh[idx], xlow[sidx], xhigh[sidx]),
#     #         #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#     #         # bot_low, bot_high = abs_bwd_phi((lA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :], uA[sidx * n_node_features_pre:sidx * n_node_features_pre + n_node_features_pre, :, :]),
#     #         #                                 (None, None, xlow[sidx], xhigh[sidx]),
#     #         #                                 (lA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :], uA[tidx * n_node_features_pre:tidx * n_node_features_pre + n_node_features_pre, :, :]))
#     #         # msg_low, msg_high = least_upper_bound(m_low, m_high, bot_low, bot_high)
#     #     messages_low[index_targets[idx]].append(msg_low)
#     #     messages_high[index_targets[idx]].append(msg_high)
#     #
#     # embeddings_low = torch.concat(
#     #     [abs_bwd_sigma(m, lA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_low)])
#     #
#     # embeddings_high = torch.concat(
#     #     [abs_bwd_sigma(m, uA[i * n_node_features_pre:i * n_node_features_pre + n_node_features_pre, :, :]) for i, m in enumerate(messages_high)])
#     #
#     # lAx = embeddings_low
#     # uAx = embeddings_high
#     def get_node_values(_node):
#         edge_idx = np.where(index_targets == _node)[0]
#         indices, lows, highs = {}, {}, {}
#         for i in range(n_node_features_pre):
#             indices[i] = []
#             lows[i] = []
#             highs[i] = []
#             for idx in edge_idx:
#                 sidx = index_sources[idx]
#                 newmlow, newmhigh = abs_phi_prod2((elow[idx], ehigh[idx], xlow[sidx][i], xhigh[sidx][i]))
#                 indices[i].append(sidx)
#                 lows[i].append(newmlow)
#                 highs[i].append(newmhigh)
#         def get_node_values_by_nonzero_indices(non_zero_idx):
#             for j in [nnz.item() % n_node_features_pre for nnz in non_zero_idx]:
#                 values_low = torch.zeros((1, n_vars_pre))
#                 values_high = torch.zeros((1, n_vars_pre))
#                 for _idx, low, high in zip(indices[j], lows[j], highs[j]):
#                     values_low[0][_idx * n_node_features_pre + j] = low
#                     values_high[0][_idx * n_node_features_pre + j] = high
#                 yield values_low, values_high
#         return get_node_values_by_nonzero_indices
#
#     reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
#     # reshaped_lax = lAx.view((n_vars_pre, 1, -1))
#     reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
#     # reshaped_uax = uAx.view((n_vars_pre, 1, -1))
#
#     lA = torch.empty_like(last_lA)
#     uA = torch.empty_like(last_uA)
#     for node in range(n_nodes):
#         node_vals_by_nnz = get_node_values(node)
#         for v_offset in range(n_node_features_post):
#             v = node * n_node_features_post + v_offset
#             print(v, 'out of', n_vars_post)
#             non_zero_indices = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
#             # node = v // n_node_features_post
#             # subvar = v % n_node_features_pre
#             # edge_idx = np.where(index_targets == node)[0]
#             if non_zero_indices.nelement() != 0:
#                 prev_values_la = reshaped_last_la[v][0][non_zero_indices]
#                 prev_values_ua = reshaped_last_ua[v][0][non_zero_indices]
#                 # values_la, values_ua = get_node_values(node)
#                 # values_la = (torch.ones((1, n_vars_pre)) for _ in range(n_vars_pre)) # torch.unbind(reshaped_lax[non_zero_indices])
#                 # values_ua = (torch.ones((1, n_vars_pre)) for _ in range(n_vars_pre)) # torch.unbind(reshaped_uax[non_zero_indices])
#                 node_vals = node_vals_by_nnz(non_zero_indices)
#                 rec_value_la = reshaped_last_la[v]
#                 rec_value_ua = reshaped_last_ua[v]
#                 emb_low, emb_high = 0., 0.
#                 for coeff_la, coeff_ua, (mat_la, mat_ua) in zip(prev_values_la, prev_values_ua, node_vals):
#                     if coeff_la >= 0 and coeff_ua >= 0:
#                         msg_low, msg_high = mat_la * coeff_la, mat_ua * coeff_ua
#                     elif coeff_la <= 0 and coeff_ua <= 0:
#                         msg_low, msg_high = mat_ua * coeff_ua, mat_la * coeff_la
#                     elif coeff_la < 0 and coeff_ua > 0:
#                         msg_low, msg_high = least_upper_bound(mat_la * coeff_la, mat_la * coeff_la, mat_ua * coeff_ua, mat_ua * coeff_ua)
#                     elif coeff_la > 0 and coeff_ua < 0:
#                         msg_low, msg_high = least_upper_bound(mat_la * coeff_la, mat_la * coeff_la, mat_ua * coeff_ua, mat_ua * coeff_ua)
#                     else:
#                         raise NotImplementedError
#                         # msg_low, msg_high = abs_phi((mat_la, mat_ua), (coeff_la, coeff_ua), (rec_value_la, rec_value_ua))
#                     emb_low, emb_high = abs_sigma([(emb_low, emb_high), (msg_low, msg_high)], (rec_value_la, rec_value_ua))
#                 embeddings_low, embeddings_high = emb_low, emb_high
#             else:
#                 embeddings_low = torch.zeros((1, n_vars_pre))# torch.zeros_like(reshaped_lax[0])
#                 embeddings_high = torch.zeros((1, n_vars_pre)) # torch.zeros_like(reshaped_uax[0])
#             lA[v] = embeddings_low.view((1, n_nodes, n_node_features_pre))
#             uA[v] = embeddings_high.view((1, n_nodes, n_node_features_pre))
#
#     lAx = lA
#     uAx = uA
#
#     return [(lAx, uAx), (None, None), (None, None)], 0., 0.

############## PREIMAGE
class Pre(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, phis, sigmas, use_optimized_gcn, hops_left):
        """ In this function, define the arguments and attributes of the operator.
        "custom::SigmaSum" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Pre', x, a, e, phi_s=phis, sigma_s=sigmas, use_optimized_gcn_s=str(use_optimized_gcn), hops_left_i=hops_left).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, phis, sigmas, use_optimized_gcn, hops_left):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = i * e in this case. """
        return concrete_message_passing(0, 1, fucts[phis], fucts[sigmas], use_optimized_gcn, x, a, e)


class PreImage(nn.Module):
    def __init__(self, phi_f, sigma_f, use_optimized_gcn, hops_left):
        super().__init__()
        self.phi_f = phi_f
        self.sigma_f = sigma_f
        self.use_optimized_gcn = use_optimized_gcn
        self.hops_left = hops_left

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Pre.apply(x, a, e, self.phi_f, self.sigma_f, self.use_optimized_gcn, self.hops_left)


class BoundPre(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.phi_f = attr['phi']
        self.sigma_f = attr['sigma']
        self.use_optimized_gcn = attr['use_optimized_gcn'] == 'True'
        self.hops_left = attr['hops_left']

    def forward(self, x, a, e):
        return concrete_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # abs_fwd_phi = fucts[self.phi_f][2]
        # abs_fwd_sigma = fucts[self.sigma_f][2]
        # n_nodes = x.lb.shape[1]
        # n_edges = e.lb.shape[1]
        # index_targets = a.lb[0][1]  # Nodes receiving the message
        # index_sources = a.lb[0][0]  # Nodes sending the message (ie neighbors)
        # e = e.lb[0]
        # lwx, uwx = x.lw[0], x.uw[0]
        # emb_list_l = []
        # emb_list_u = []
        # for k in range(dim_in):
        #     lw = lwx[k]
        #     uw = uwx[k]
        #     messages_l = [[] for _ in range(n_nodes)]  # list of lists of messages
        #     messages_u = [[] for _ in range(n_nodes)]
        #     for idx in range(n_edges):
        #         sidx = index_sources[idx]
        #         tidx = index_targets[idx]
        #         messages_l[index_targets[idx]].append(abs_fwd_phi(lw[sidx], e[idx], lw[tidx]))
        #         messages_u[index_targets[idx]].append(abs_fwd_phi(uw[sidx], e[idx], uw[tidx]))
        #
        #     embeddings_l = torch.stack([abs_fwd_sigma(m, lw[i]) for i, m in enumerate(messages_l)])
        #     embeddings_u = torch.stack([abs_fwd_sigma(m, uw[i]) for i, m in enumerate(messages_u)])
        #     emb_list_l.append(embeddings_l)
        #     emb_list_u.append(embeddings_u)
        #
        # embeddings_l = torch.stack(emb_list_l).unsqueeze(0)
        # embeddings_u = torch.stack(emb_list_u).unsqueeze(0)
        #
        # return LinearBound(lw=embeddings_l, lb=x.lb, uw=embeddings_u, ub=x.ub)

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        # def update_shapes(node, reach):
        #     node.upper = node.upper[:, reach, :]
        #     node.shape = node.upper.shape[1:]
        #     node.output_shape = node.upper.shape
        #     node.lower = node.lower[:, reach, :]
        #     node.interval = node.interval[0][:, reach, :], node.interval[1][:, reach, :]
        #     node.input_shape = node.output_shape
        #     node.flattened_nodes = torch.prod(torch.tensor(node.shape))
        #     for inp in node.inputs:
        #         if not isinstance(inp, BoundInput) and not isinstance(inp, BoundTranspose):
        #             update_shapes(inp, reach)
        #         elif isinstance(inp, BoundInput) and not isinstance(inp, BoundParams):
        #             if inp.name != '/x.1': continue
        #             inp.value = BoundedTensor(inp.value[:, reach, :], inp.value.ptb)
        #             inp.upper = inp.upper[:, reach, :]
        #             inp.perturbation = inp.value.ptb
        #             inp.output_shape = inp.value.shape
        #             inp.lower = inp.lower[:, reach, :]
        #             inp.linear.lower = inp.lower
        #             inp.linear.upper = inp.upper
        #             inp.interval = Interval(inp.interval[0][:, node_reachability, :], inp.interval[1][:, node_reachability, :], inp.perturbation)
        #             inp.center = inp.center[:, node_reachability, :]
        #
        #
        # # if self.graph_abstraction == 'NoAbstraction' or self.graph_abstraction == 'BisimAbstraction':
        #
        # if self.hops_left > 0 and True:
        #     node = 1
        #     node_reachability = sum(torch.linalg.matrix_power(a.forward_value, i) for i in range(self.hops_left + 1))[0][:, node] > 0
        #
        #     a.value = a.value[:, node_reachability, :][:, :, node_reachability]
        #     a.upper = a.upper[:, node_reachability, :][:, :, node_reachability]
        #     a.output_shape = a.value.shape
        #     a.lower = a.lower[:, node_reachability, :][:, :, node_reachability]
        #     a.interval = a.value, a.value
        #     a.forward_value = a.value
        #     a.center = a.value
        #
        #     e.value = BoundedTensor(e.value[:, node_reachability, :][:, :, node_reachability], PerturbationLpNorm(x_L=e.value.ptb.x_L[:, node_reachability, :][:, :, node_reachability], x_U=e.value.ptb.x_U[:, node_reachability, :][:, :, node_reachability]))
        #     e.upper = e.upper[:, node_reachability, :][:, :, node_reachability]
        #     e.perturbation = PerturbationLpNorm(x_L=e.value.ptb.x_L, x_U=e.value.ptb.x_U)
        #     e.output_shape = e.value.shape
        #     e.lower = e.lower[:, node_reachability, :][:, :, node_reachability]
        #     e.linear.lower = e.lower
        #     e.linear.upper = e.upper
        #     e.interval = Interval(e.interval[0][:, node_reachability, :][:, :, node_reachability], e.interval[1][:, node_reachability, :][:, :, node_reachability], e.perturbation)
        #     e.center = e.center[:, node_reachability, :][:, :, node_reachability]
        #
        #     x.upper = x.upper[:, node_reachability, :]
        #     x.shape = x.upper.shape[1:]
        #     x.output_shape = x.upper.shape
        #     x.lower = x.lower[:, node_reachability, :]
        #     x.interval = x.interval[0][:, node_reachability, :], x.interval[1][:, node_reachability, :]
        #     x.input_shape = x.output_shape
        #     x.flattened_nodes = torch.prod(torch.tensor(x.shape))
        #     update_shapes(x.inputs[0], node_reachability)
        #
        #
        #     node_indices = torch.where(node_reachability)[0]
        #     node_features_post = last_lA.shape[0] // last_lA.shape[2]
        #     matrix_indices = torch.cat([torch.tensor([node_features_post * i, node_features_post * i + 1]) for i in node_indices])
        #
        #
        #     la_filtered = lAx[matrix_indices]
        #     lAx = la_filtered[:, :, node_reachability, :]
        #
        #     ua_filtered = uAx[matrix_indices]
        #     uAx = ua_filtered[:, :, node_reachability, :]
        #
        #     lae_filtered = lAe[matrix_indices]
        #     lAe = lae_filtered[:, :, node_reachability, :][:, :, :, node_reachability]
        #
        #     uae_filtered = uAe[matrix_indices]
        #     uAe = uae_filtered[:, :, node_reachability, :][:, :, :, node_reachability]
        #
        #     lbias = lbias[matrix_indices]
        #
        #     ubias = ubias[matrix_indices]
        # elif self.hops_left == 0 and True:
        #     node = 1
        #     node_reachability = sum(torch.linalg.matrix_power(a.forward_value, i) for i in range(self.hops_left + 1))[0][:, node] > 0
        #     node_reachability = torch.Tensor([True, True, True, False, True])
        #     node_indices = torch.where(node_reachability)[0]
        #     node_features_post = last_lA.shape[0] // last_lA.shape[2]
        #     matrix_indices = torch.cat([torch.tensor([node_features_post * i, node_features_post * i + 1]) for i in node_indices])
        #     x.reachability = matrix_indices

        ((lAx, uAx), _, (lAe, uAe)), lbias, ubias = bwd_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, last_lA, last_uA, x, a, e)
        return [(lAx, uAx), (None, None), (lAe, uAe)], lbias, ubias
        # elif self.graph_abstraction == 'EdgeAbstraction':
        #     return bwd_message_passing_edge_abs(0, 1, fucts[self.phi_f], fucts[self.sigma_f], last_lA, last_uA, x, a, e)
        # else:
        #     raise NotImplementedError

    def interval_propagate(self, *v):
        """ IBP computation """
        x, a, e = v
        # if self.graph_abstraction == 'NoAbstraction' or self.graph_abstraction == 'BisimAbstraction':
        return interval_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)
        # elif self.graph_abstraction == 'EdgeAbstraction':
        #     return interval_message_passing_edge_abs(0, 1, fucts[self.phi_f], fucts[self.sigma_f], x, a, e)
        # else:
        #     raise NotImplementedError


register_custom_op("custom::Pre", BoundPre)

########################################################################################################################################################

############## POSTIMAGE
class Post(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, define the arguments and attributes of the operator.
        "custom::SigmaSum" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Post', x, a, e, phi_s=phis, sigma_s=sigmas, use_optimized_gcn_s=str(use_optimized_gcn)).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = i * e in this case. """
        return concrete_message_passing(1, 0, fucts[phis], fucts[sigmas], use_optimized_gcn, x, a, e)


class PostImage(nn.Module):
    def __init__(self, phi_f, sigma_f, use_optimized_gcn):
        super().__init__()
        self.phi_f = phi_f
        self.sigma_f = sigma_f
        self.use_optimized_gcn = use_optimized_gcn

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Post.apply(x, a, e, self.phi_f, self.sigma_f, self.use_optimized_gcn)


class BoundPost(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.phi_f = attr['phi']
        self.sigma_f = attr['sigma']
        self.use_optimized_gcn = attr['use_optimized_gcn'] == 'True'

    def forward(self, x, a, e):
        return concrete_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # abs_fwd_phi = fucts[self.phi_f][2]
        # abs_fwd_sigma = fucts[self.sigma_f][2]
        # n_nodes = x.lb.shape[1]
        # n_edges = e.lb.shape[1]
        # index_targets = a.lb[0][1]  # Nodes receiving the message
        # index_sources = a.lb[0][0]  # Nodes sending the message (ie neighbors)
        # e = e.lb[0]
        # lwx, uwx = x.lw[0], x.uw[0]
        # emb_list_l = []
        # emb_list_u = []
        # for k in range(dim_in):
        #     lw = lwx[k]
        #     uw = uwx[k]
        #     messages_l = [[] for _ in range(n_nodes)]  # list of lists of messages
        #     messages_u = [[] for _ in range(n_nodes)]
        #     for idx in range(n_edges):
        #         sidx = index_sources[idx]
        #         tidx = index_targets[idx]
        #         messages_l[index_sources[idx]].append(abs_fwd_phi(lw[tidx], e[idx], lw[sidx]))
        #         messages_u[index_sources[idx]].append(abs_fwd_phi(uw[tidx], e[idx], uw[sidx]))
        #
        #     embeddings_l = torch.stack([abs_fwd_sigma(m, lw[i]) for i, m in enumerate(messages_l)])
        #     embeddings_u = torch.stack([abs_fwd_sigma(m, uw[i]) for i, m in enumerate(messages_u)])
        #     emb_list_l.append(embeddings_l)
        #     emb_list_u.append(embeddings_u)
        #
        # embeddings_l = torch.stack(emb_list_l).unsqueeze(0)
        # embeddings_u = torch.stack(emb_list_u).unsqueeze(0)
        #
        # return LinearBound(lw=embeddings_l, lb=x.lb, uw=embeddings_u, ub=x.ub)

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        # if self.graph_abstraction == 'NoAbstraction' or self.graph_abstraction == 'BisimAbstraction':
        return bwd_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, last_lA, last_uA, x, a, e)
        # elif self.graph_abstraction == 'EdgeAbstraction':
        #     return bwd_message_passing_edge_abs(1, 0, fucts[self.phi_f], fucts[self.sigma_f], last_lA, last_uA, x, a, e)
        # else:
        #     raise NotImplementedError

    def interval_propagate(self, *v):
        """ IBP computation """
        x, a, e = v
        # if self.graph_abstraction == 'NoAbstraction' or self.graph_abstraction == 'BisimAbstraction':
        return interval_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)
        # elif self.graph_abstraction == 'EdgeAbstraction':
        #     return interval_message_passing_edge_abs(1, 0, fucts[self.phi_f], fucts[self.sigma_f], x, a, e)
        # else:
        #     raise NotImplementedError


register_custom_op("custom::Post", BoundPost)

########################################################################################################################################################


############## ITE-CROSS
class Ite(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, b, a, e, iftrue, iffalse, shp):
        return g.op('custom::Ite', x, b, a, e, shp_t=shp, iftrue_s=mg_reconstructor.reconstruct(iftrue.expr), iffalse_s=mg_reconstructor.reconstruct(iffalse.expr)).setType(x.type())

    @staticmethod
    def forward(ctx, x, b, a, e, iftrue, iffalse, shp):
        if torch.all(b):
            return iftrue(x, a, e)
        else:
            return iffalse(x, a, e)


class Choice(nn.Module):
    def __init__(self, iftrue, iffalse):
        super().__init__()
        self.iftrue = iftrue
        self.iffalse = iffalse

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        b, x = x
        return Ite.apply(x, b, a, e, self.iftrue, self.iffalse, x.shape)


class BoundChoice(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        x, b, a, e = inputs
        self.x = x
        self.a = a
        self.e = e
        self.iftrue = BoundedModule(interpreter.run(attr['iftrue']), (torch.empty_like(torch.zeros(attr['shp'])), a.value, torch.empty_like(e.value)), device=x.device, verbose=True)
        self.iffalse = BoundedModule(interpreter.run(attr['iffalse']), (torch.empty_like(torch.zeros(attr['shp'])), a.value, torch.empty_like(e.value)), device=x.device, verbose=True)

    def forward(self, x, b, a, e):
        return torch.cond(torch.all(b), self.iftrue, self.iffalse, (x, a, e))

    def bound_forward(self, dim_in, x, b, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # lb, ub = b.lower, b.upper
        # x_in = interval_to_bounded_tensor(x.lower, x.upper)
        # if torch.all(lb): # if lower bound is all True, guaranteed iftrue branch
        #     _, _ = self.iftrue.compute_bounds((x_in, self.a.value, self.e.value), method='forward')
        #     lw, uw, lbound, ubound =  self.iftrue[self.iftrue.final_name].linear.lw, self.iftrue[self.iftrue.final_name].linear.uw, self.iftrue[self.iftrue.final_name].linear.lb, self.iftrue[self.iftrue.final_name].linear.ub
        #     return LinearBound(lw=lw, lb=lbound, uw=uw, ub=ubound)
        # elif not torch.all(ub): # if upper bound is not all True, guaranteed iffalse branch
        #     _, _ = self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='forward')
        #     lw, uw, lbound, ubound =  self.iffalse[self.iffalse.final_name].linear.lw, self.iffalse[self.iffalse.final_name].linear.uw, self.iffalse[self.iffalse.final_name].linear.lb, self.iffalse[self.iffalse.final_name].linear.ub
        #     return LinearBound(lw=lw, lb=lbound, uw=uw, ub=ubound)
        # else:
        #     tmpl, tmpu = self.iftrue.compute_bounds((x, self.a.value, self.e.value), method='forward')
        #     tlw, tuw, tlbound, tubound =  self.iftrue[self.iftrue.final_name].linear.lw, self.iftrue[self.iftrue.final_name].linear.uw, self.iftrue[self.iftrue.final_name].linear.lb, self.iftrue[self.iftrue.final_name].linear.ub
        #     fmpl, fmpu = self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='forward')
        #     flw, fuw, flbound, fubound =  self.iffalse[self.iffalse.final_name].linear.lw, self.iffalse[self.iffalse.final_name].linear.uw, self.iffalse[self.iffalse.final_name].linear.lb, self.iffalse[self.iffalse.final_name].linear.ub
        #     lw, uw = forward_lub(tlw, tuw, tmpl, tmpu, flw, fuw, fmpl, fmpu)
        #     lbound, ubound = least_upper_bound(tlbound, tubound, flbound, fubound)
        #     return LinearBound(lw=lw, lb=lbound, uw=uw, ub=ubound)

    def bound_backward(self, last_lA, last_uA, x, b, a, e, **kwargs):
        """ Backward mode bound propagation """
        lb, ub = b.lower, b.upper
        x = interval_to_bounded_tensor(x.lower, x.upper)
        if torch.all(lb): # if lower bound is all True, guaranteed iftrue branch
            _, _, A_dict = self.iftrue.compute_bounds((x, a.value, e.value), C=torch.transpose(last_lA, 0, 1), method='backward', return_A=True, need_A_only=True, needed_A_dict={self.iftrue.final_name: [self.iftrue.final_name]})
            lA, uA = A_dict[self.iftrue.final_name][self.iftrue.final_name]['lA'], A_dict[self.iftrue.final_name][self.iftrue.final_name]['uA']
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0
        elif not torch.all(ub): # if upper bound is not all True, guaranteed iffalse branch
            _, _, A_dict = self.iffalse.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward', return_A=True, need_A_only=True, needed_A_dict={self.iffalse.final_name: [self.iffalse.final_name]})
            lA, uA = A_dict[self.iffalse.final_name][self.iffalse.final_name]['lA'], A_dict[self.iffalse.final_name][self.iffalse.final_name]['uA']
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0
        else:
            tmpl, tmpu, A_dict = self.iftrue.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward', return_A=True, need_A_only=False, needed_A_dict={self.iftrue.final_name: [self.iftrue.final_name]})
            tlA, tuA = A_dict[self.iftrue.final_name][self.iftrue.final_name]['lA'], A_dict[self.iftrue.final_name][self.iftrue.final_name]['uA']
            fmpl, fmpu, A_dict = self.iffalse.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward', return_A=True, need_A_only=False, needed_A_dict={self.iffalse.final_name: [self.iffalse.final_name]})
            flA, fuA = A_dict[self.iffalse.final_name][self.iffalse.final_name]['lA'], A_dict[self.iffalse.final_name][self.iffalse.final_name]['uA']
            lA, uA = backward_lub(tlA, tuA, tmpl, tmpu, flA, fuA, fmpl, fmpu)
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0

    def interval_propagate(self, *v):
        """ IBP computation """
        x, b, a, e = v
        lb, ub = b
        x = interval_to_bounded_tensor(*x)
        if torch.all(lb): # if lower bound is all True, guaranteed iftrue branch
            return self.iftrue.compute_bounds((x, self.a.value, self.e.value), method='IBP')
        elif not torch.all(ub): # if upper bound is not all True, guaranteed iffalse branch
            return self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='IBP')
        else: # lub of iftrue and iffalse
            ift = self.iftrue.compute_bounds((x, self.a.value, self.e.value), method='IBP')
            iff = self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='IBP')
            tlb, tub = ift[0], ift[1]
            flb, fub = iff[0], iff[1]
            return least_upper_bound(tlb, tub, flb, fub)


register_custom_op("custom::Ite", BoundChoice)

########################################################################################################################################################

############## FIX
class Fix(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, body, atol, rtol, shp):
        return g.op('custom::Fix', x, a, e, shp_t=shp, body_s=mg_reconstructor.reconstruct(body.expr), atol_f=atol, rtol_f=rtol).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, body, atol, rtol, shp):
        x_old = x
        x = body(x_old, a, e)
        while not torch.allclose(x_old, x, atol=atol, rtol=rtol) and not torch.any(torch.isnan(x)):
            x_old = x
            x = body(x_old, a, e)
        return x


class FixPoint(nn.Module):
    def __init__(self, body, atol, rtol):
        super().__init__()
        self.body = body
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Fix.apply(x, a, e, self.body, self.atol, self.rtol, x.shape)


class BoundFix(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        x, a, e = inputs
        self.x = x
        self.a = a
        self.e = e
        self.atol = attr['atol']
        self.rtol = attr['rtol']
        interpreter.set_tolerance(self.atol, self.rtol)
        self.body = BoundedModule(interpreter.run(attr['body']), (torch.empty_like(torch.zeros(attr['shp'])), a.value, torch.empty_like(e.value)), device=x.device, verbose=True)

    def forward(self, x, a, e):
        x_old = x
        x = self.body(x_old, a, e)
        while not torch.allclose(x_old, x, atol=self.atol, rtol=self.rtol) and not torch.any(torch.isnan(x)):
            x_old = x
            x = self.body(x_old, a, e)
        return x

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # R = x.lower, x.upper
        # T = R
        # R = self.body.compute_bounds((self.x.value, self.a.value, self.e.value), method='forward')
        # R = least_upper_bound(*T, *R)
        # while not torch.allclose(T[0], R[0], atol=self.atol, rtol=self.rtol) and not torch.allclose(T[1], R[1], atol=self.atol, rtol=self.rtol):
        #     T = R
        #     ptb = PerturbationLpNorm(norm=np.inf, x_L=R[0], x_U=R[1])
        #     self.body[self.body.root_names[0]].perturbation = ptb
        #     self.body[self.body.root_names[0]].linear.lw = self.body[self.body.final_name].linear.lw
        #     self.body[self.body.root_names[0]].linear.lb = self.body[self.body.final_name].linear.lb
        #     self.body[self.body.root_names[0]].linear.lower = R[0]
        #     self.body[self.body.root_names[0]].lower = R[0]
        #     self.body[self.body.root_names[0]].linear.uw = self.body[self.body.final_name].linear.uw
        #     self.body[self.body.root_names[0]].linear.ub = self.body[self.body.final_name].linear.ub
        #     self.body[self.body.root_names[0]].linear.upper = R[1]
        #     self.body[self.body.root_names[0]].upper = R[1]
        #     self.body[self.body.root_names[0]].interval = R
        #     C = torch.eye(dim_in, device=self.device).expand(1, dim_in, dim_in)
        #     R = self.body.forward_general(C=C, node=self.body[self.body.final_name], concretize=True)
        #     R = least_upper_bound(*T, *R)
        # lw, uw, lbound, ubound =  self.body[self.body.final_name].linear.lw, self.body[self.body.final_name].linear.uw, self.body[self.body.final_name].linear.lb, self.body[self.body.final_name].linear.ub
        # return LinearBound(lw=lw, lb=lbound, uw=uw, ub=ubound)

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        R = x.interval
        T = R
        R = interval_to_bounded_tensor(*R)
        *R, A_dict = self.body.compute_bounds((R, a.value, e.value), method='backward', C=torch.transpose(last_lA, 0, 1), return_A=True, needed_A_dict={self.body.final_name: [self.body.final_name]})
        R = least_upper_bound(*T, *R)
        while not torch.allclose(T[0], R[0], atol=self.atol, rtol=self.rtol) and not torch.allclose(T[1], R[1], atol=self.atol, rtol=self.rtol):
            T = R
            R = interval_to_bounded_tensor(*R)
            *R, A_dict = self.body.compute_bounds((R, a.value, e.value), C=A_dict[self.body.final_name][self.body.final_name]['lA'], method='backward', return_A=True, needed_A_dict={self.body.final_name: [self.body.final_name]})
            R = least_upper_bound(*T, *R)
        lA, uA = A_dict[self.body.final_name][self.body.final_name]['lA'], A_dict[self.body.final_name][self.body.final_name]['uA']
        return [(torch.transpose(lA, 0 ,1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0

    def interval_propagate(self, *v):
        """ IBP computation """
        R, a, e = v
        T = R
        R = interval_to_bounded_tensor(*R)
        # R = least_upper_bound(*R, *self.body.compute_bounds((self.x.value, self.a.value, self.e.value), method='IBP'))
        R = self.body.compute_bounds((R, self.a.value, self.e.value), method='IBP')
        while not torch.allclose(T[0], R[0], atol=self.atol, rtol=self.rtol) and not torch.allclose(T[1], R[1], atol=self.atol, rtol=self.rtol):
            T = R
            R = interval_to_bounded_tensor(*R)
            # R = least_upper_bound(*R, *self.body.compute_bounds((new_x, self.a.value, self.e.value), method='IBP'))
            R = self.body.compute_bounds((R, self.a.value, self.e.value))
        return R


register_custom_op("custom::Fix", BoundFix)

########################################################################################################################################################
class PreGCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, a, e):
        return torch.matmul(e.transpose(-1, -2), x)


class PostGCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, a, e):
        return torch.matmul(e, x)



# Abstract interpreter
class LirpaInterpreter(Interpreter):
    def __init__(self):
        self.graph_abstraction = None
        self.atol = 0.000001
        self.rtol = 0
        self.mg_layers = None
        self.context = Context()
        self.use_optimized_gcn = None
        self.hops = 0
        self.rev_hops = 0

    def set_graph_abstraction(self, graph_abstraction):
        self.graph_abstraction = graph_abstraction

    def set_concrete_layers(self, mg_layers):
        self.mg_layers = mg_layers

    def set_tolerance(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol

    # def optimized_gcn(self, value):
    #     self.use_optimized_gcn = value

    def get_concrete_layer(self, tree):
        return self.mg_layers[hash(self.context.get(tree))]

    def get_abstract_layer(self, conc_op):
        psi = conc_op.psi
        if isinstance(psi.f, layers.Dense):  # Dense layer
            layer = psi.f
            return make_layer(layer.trainable_variables[0].value.numpy(), layer.trainable_variables[1].value.numpy(), layer.activation.__name__)
        else: # Pooling layer
            return make_pooling('sum' if psi.fname == 'SumPooling' else 'mean', self.graph_abstraction)


    def run(self, expr):  # inputs in tf format
        tree = mg_parser.parse(expr) if isinstance(expr, str) else expr
        output = self.visit(tree)
        return output

    @v_args(inline=True)
    def label(self, tree):
        return str(tree)

    @v_args(inline=True)
    def id(self):
        return torch.nn.Identity()

    def atom_op(self, tree):
        concrete_op = self.get_concrete_layer(tree)
        abstract_op = self.get_abstract_layer(concrete_op)

        class Atom(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)
            def forward(self, x, a, e):
                for layer in self.layers:
                    x = layer(x)
                return x

        if isinstance(abstract_op, list) or isinstance(abstract_op, tuple):
            return Atom(abstract_op)
        else:
            return Atom([abstract_op])

    def lhd(self, tree):
        phi, sigma = tree.children
        phi, sigma = self.visit(phi), self.visit(sigma)
        concrete_op = self.get_concrete_layer(tree)
        hops = self.rev_hops
        self.rev_hops += 1
        return PreImage(phi, sigma, self.graph_abstraction.optimized_gcn, hops)

    def rhd(self, tree):
        phi, sigma = tree.children
        phi, sigma = self.visit(phi), self.visit(sigma)
        concrete_op = self.get_concrete_layer(tree)
        return PostImage(phi, sigma, self.graph_abstraction.optimized_gcn)

    def sequential_composition(self, tree):
        left, right = tree.children
        class Sequential(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)
            def forward(self, x, a, e):
                for layer in self.layers:
                    x = layer(x, a, e)
                return x
        phi = self.visit(left)
        self.context.push(left)
        psi = self.visit(right)
        self.context.pop()
        return Sequential([phi, psi])

    @v_args(inline=True)
    def parallel_composition(self, left, right):
        class Parallel(torch.nn.Module):
            def __init__(self, l, r):
                super().__init__()
                self.l = l
                self.r = r
            def forward(self, x, a, e):
                y = self.l(x, a, e)
                z = self.r(x, a, e)
                return y, z
        left = self.visit(left)
        right = self.visit(right)
        return Parallel(left, right)

    @v_args(inline=True)
    def choice(self, left, right):
        iftrue = self.visit(left)
        iftrue.expr = left
        iffalse = self.visit(right)
        iffalse.expr = right
        return Choice(iftrue, iffalse)

    @v_args(inline=True)
    def star(self, body):
        loop = self.visit(body)
        loop.expr = body
        return FixPoint(loop, self.atol, self.rtol)


interpreter = LirpaInterpreter()



# def copier(value):
#     if isinstance(value, tuple) and isinstance(value[0], list):
#         perturbation = value[0][0].ptb
#         tensors = tuple([BoundedTensor(deepcopy(t[0].data), perturbation)] + t[1:] for t in value)
#         return tensors
#     else:
#         return deepcopy(value)


# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PostImage('phi_prod', 'sigma_sum')
#         self.lin1 = torch.nn.Linear(3, 5, bias=False)
#         self.relu = torch.nn.ReLU()
#         self.pre2 = PostImage('phi_prod', 'sigma_sum')
#         self.lin2 = torch.nn.Linear(5, 2, bias=False)
#         self.sfmax = torch.nn.Softmax()
#
#     def forward(self, x, a, e):
#         x1 = self.pre1(x, a, e)
#         x2 = self.lin1(x1)
#         x3 = self.relu(x2)
#         x4 = self.pre2(x3, a, e)
#         x5 = self.lin2(x4)
#         return x5

# PAR
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.post1 = PostImage('phi_prod', 'sigma_sum')
#
#     def forward(self, x, a, e):
#         x1 = self.pre1(x, a, e)
#         x2 = self.post1(x, a, e)
#         return torch.add(x1, x2)


# ITE
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.post1 = PostImage('phi_prod', 'sigma_sum')
#         self.ite = Choice(self.pre1, self.post1)
#
#     def forward(self, x, xb, a, e):
#         x1 = self.ite(x, xb, a, e)
#         return x1

# FIX
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.fix = FixPoint(self.pre1, 0.000001, 0)
#
#     def forward(self, x, a, e):
#         x1 = self.fix(x, a, e)
#         return x1

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre1 = PostImage('x', '+')


    def forward(self, x, a, e):
        return torch.matmul(e, x)


def check_soundness(pred, lb, ub):
    eps = interpreter.atol
    pred = pred[0] if isinstance(pred, tuple) else pred
    pred = pred[0] if pred.shape.ndims == 3 else pred
    lb = lb[0] if lb.ndim == 3 else lb
    ub = ub[0] if ub.ndim == 3 else ub
    for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
        for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
            assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
    print('Soundness check passed')


if __name__ == '__main__':
    expr = '<x|+ ; lin ; <x|+ ; out'
    channels = 5

    x_in = torch.tensor([[[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]]], dtype=torch.float32)
    bool_in = torch.tensor([[[1], [1], [1], [1], [1]]], dtype=torch.float32)
    a_in = torch.tensor([[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], [0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 1, 4]]])
    w_in = torch.tensor([[[0.2, 0.1, 0.4, 0, 0], [0, 0.1, 0.2, 0.3, 0], [0, 0.2, 0.5, 0.7, 0], [0, 0, 0, 0.4, 0.5], [0, 0.1, 0, 0, 0.5]]])
    e_in = torch.tensor([[0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.3333333], [0.40824828], [0.3333333], [0.3333333], [0.40824828], [0.49999997],
                          [0.49999997], [0.40824828], [0.49999997]], dtype=torch.float32)
    mat = torch.tensor([[[0.3333333, 0.3333333, 0.3333333, 0, 0], [0, 0.3333333, 0.3333333, 0.40824828, 0], [0, 0.3333333, 0.3333333, 0.40824828, 0],
                        [0, 0, 0, 0.49999997, 0.49999997], [0, 0.40824828, 0, 0, 0.49999997]]], dtype=torch.float32)

    model = MyModel()

    lirpa_model = BoundedModule(model, (torch.empty_like(x_in), a_in, torch.empty_like(mat)), device=x_in.device, verbose=True)
    # lirpa_model = BoundedModule(model, (torch.empty_like(x_in), torch.empty_like(bool_in), a_in, torch.empty_like(e_in)), device=x_in.device, verbose=True)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # bool_ptb = PerturbationLpNorm(norm=np.inf, x_L=bool_in-1, x_U=bool_in)
    my_input = BoundedTensor(x_in, ptb)
    e_input = BoundedTensor(mat, ptb)
    # my_input_bool = BoundedTensor(bool_in, bool_ptb)


    # prediction = model(my_input, my_input_bool, a_in, e_in)
    prediction = model(my_input, a_in, e_input)

    print(prediction)

    abs_prediction_2 = lirpa_model.compute_bounds(x=(my_input, a_in, e_input), method='backward', IBP=True)

    # abs_prediction_2 = lirpa_model.compute_bounds((my_input, my_input_bool, a_in, e_in), method='forward')

    # abs_prediction_2 = lirpa_model.compute_bounds(x=(my_input, my_input_bool, a_in, e_in))
    #
    print(abs_prediction_2)
    #
    abs_lb, abs_ub = abs_prediction_2
    #
    check_soundness(tf.constant(prediction.detach().numpy()), tf.constant(abs_lb.detach().numpy()), tf.constant(abs_ub.detach().numpy()))


