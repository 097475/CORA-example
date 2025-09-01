from copy import deepcopy

import torch
from lark.visitors import Interpreter, v_args
from libmg import mg_parser

from forward_abstract_interpretation.lirpa_domain import PreImage, PostImage, Choice


class AbstractMemory:
    def __init__(self, x, a, e):
        self.x = (x, ) if not isinstance(x, tuple) else x
        self.a = a
        self.e = e

    def psi(self, abs_apply, abs_psi):
        new_value = abs_apply(abs_psi, self.x)
        self.update(new_value)

    def pre(self, abs_pre, abs_phi, abs_sigma):
        new_value = abs_pre(abs_phi, abs_sigma, self.x, self.a, self.e)
        self.update(new_value)

    def post(self, abs_post, abs_phi, abs_sigma):
        new_value = abs_post(abs_phi, abs_sigma, self.x, self.a, self.e)
        self.update(new_value)

    def clone(self, copier):
        x_copy = copier(self.x)
        a_copy = copier(self.a)
        e_copy = copier(self.e)
        return AbstractMemory(x_copy, a_copy, e_copy)

    def update(self, new_x):
        self.x = (new_x, ) if not isinstance(new_x, tuple) else new_x


def meta_memory_propagate(f, _data, children, meta):
    # print(_data)
    if _data not in {'label'}:
        for c in children:
            c.meta.mem = meta.mem
        data = meta.mem
    else:
        data = None
    return f(data, *children)


@v_args(wrapper=meta_memory_propagate)
class AbstractInterpreter(Interpreter):
    def __init__(self, domain, abs_psi, abs_phi, abs_sigma):
        self.domain = domain
        self.psi = abs_psi
        self.phi = abs_phi
        self.sigma = abs_sigma
        self.history = []
        self.save_history = False

    def run(self, x, a, e, expr, save_history=False):  # inputs in tf format
        tree = mg_parser.parse(expr) if isinstance(expr, str) else expr
        tree.meta.mem = AbstractMemory(x, a, e)

        self.save_history = save_history
        if self.save_history:
            self.history = [tree.meta.mem.clone(self.domain.copier)]

        self.visit(tree)
        output = tree.meta.mem
        return output

    def get_labels(self, program_point):
        if program_point > len(self.history) + 1:
            raise ValueError("No labels at program point: ", program_point)
        return self.history[program_point]

    def label(self, _, tree):
        return str(tree)

    def atom_op(self, mem, op):
        op = self.visit(op)
        psi = self.psi[op]
        mem.psi(self.domain.abs_apply, psi)
        if self.save_history:
            self.history.append(mem.clone(self.domain.copier))

    def lhd(self, mem, phi, sigma):
        phi, sigma = self.visit(phi), self.visit(sigma)
        phi = self.phi[phi]
        sigma = self.sigma[sigma]
        mem.pre(self.domain.abs_pre, phi, sigma)
        if self.save_history:
            self.history.append(mem.clone(self.domain.copier))

    def rhd(self, mem, phi, sigma):
        phi, sigma = self.visit(phi), self.visit(sigma)
        phi = self.phi[phi]
        sigma = self.sigma[sigma]
        mem.post(self.domain.abs_post, phi, sigma)
        if self.save_history:
            self.history.append(mem.clone(self.domain.copier))

    def sequential_composition(self, _, left, right):
        self.visit(left)
        self.visit(right)

    def parallel_composition(self, mem, left, right):
        left.meta.mem = mem.clone(self.domain.copier)
        right.meta.mem = mem.clone(self.domain.copier)
        self.visit(left)
        self.visit(right)
        mem.update(left.meta.mem.x + right.meta.mem.x)





