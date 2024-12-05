from copy import deepcopy
from lark.visitors import Interpreter, v_args
from libmg import mg_parser

class AbstractMemory:
    def __init__(self, x, a, e):
        self.x = (x, )
        self.a = a
        self.e = e

    def psi(self, abs_apply, abs_psi):
        new_value = abs_apply(abs_psi, self.x)
        self.update(new_value)

    def pre(self, abs_pre, abs_phi, abs_sigma):
        new_value = abs_pre(abs_phi, abs_sigma, self.x, self.a, self.e)
        self.update(new_value)

    def clone(self):
        return deepcopy(self)

    def update(self, new_x):
        self.x = (new_x, )


def meta_memory_propagate(f, _data, children, meta):
    print(_data)
    if _data not in {'label'}:
        for c in children:
            c.meta.mem = meta.mem
        data = meta.mem
    else:
        data = None
    return f(data, *children)


@v_args(wrapper=meta_memory_propagate)
class AbstractInterpreter(Interpreter):
    def __init__(self, domain, forward, abs_psi, abs_phi, abs_sigma):
        self.domain = domain
        self.forward = True if forward in {'fw', 'forward'} else False
        self.psi = abs_psi
        self.phi = abs_phi
        self.sigma = abs_sigma
        self.history = []
        self.save_history = False

    def run(self, x, a, e, expr, save_history=False):  # inputs in tf format
        tree = mg_parser.parse(expr)
        tree.meta.mem = AbstractMemory(x, a, e)

        self.save_history = save_history
        if self.save_history:
            self.history = [tree.meta.mem.clone()]

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
            self.history.append(mem.clone())

    def lhd(self, mem, phi, sigma):
        phi, sigma = self.visit(phi), self.visit(sigma)
        phi = self.phi[phi]
        sigma = self.sigma[sigma]
        mem.pre(self.domain.abs_pre, phi, sigma)
        if self.save_history:
            self.history.append(mem.clone())

    def sequential_composition(self, _, left, right):
        if self.forward:
            self.visit(left)
            self.visit(right)
        else:
            self.visit(right)
            self.visit(left)


    # def rhd(self, meta, phi, sigma):
    #     phi, sigma = self.visit(phi), self.visit(sigma)
    #     phi = self.phi[phi]
    #     sigma = self.sigma[sigma]
    #
    #     # Message
    #     messages = [[] for _ in range(self.n_nodes)]  # list of lists of messages
    #     for src, label, tgt in zip(self.index_targets, self.e, self.index_sources):
    #         messages[tgt].append(phi(*self.get_node_labels(src), label, *self.get_node_labels(tgt)))
    #     # Aggregate
    #     embeddings = list(map(lambda x: sigma(x[1], *self.get_node_labels(x[0])).tolist()[0], enumerate(messages)))
    #     # Update
    #     self.x = (iv.matrix(embeddings), )
    #     meta.post = True
    #     if self.save_history:
    #         self.history.append(self.x)

