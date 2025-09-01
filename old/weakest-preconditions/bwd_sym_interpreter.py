from lark.visitors import Interpreter, v_args, visit_children_decor
from libmg import mg_parser
from z3 import *
set_option(precision=3)
set_option(rational_to_decimal=True)


class BWDSYMInterpreter(Interpreter):
    def __init__(self, abs_psi, abs_phi, abs_sigma):
        self.psi = abs_psi
        self.phi = abs_phi
        self.sigma = abs_sigma
        self.conditions = []
        self.past_candidates = set()
        self.current_candidate = None
        self.pre = None
        self.post = None
        self.array_ref = None
        self.n_node_features = None
        self.a = None
        self.e = None
        self.index_sources = None
        self.index_targets = None
        self.n_nodes = None


    def run(self, array_ref, precondition, postcondition, n_node_features, a, e, expr, final_bound, node):  # inputs in tf format
        self.final_bound = final_bound[0]
        self.array_ref = array_ref
        self.node = node
        self.a = a
        self.e = e.numpy()
        self.n_nodes = self.a.shape[0]
        self.index_targets = self.a.indices[:, 1].numpy()  # Nodes receiving the message
        self.index_sources = self.a.indices[:, 0].numpy()  # Nodes sending the message (ie neighbors)
        self.conditions = [postcondition]
        self.past_candidates = set()
        self.abort = False

        while not self.abort:
            print(len(self.past_candidates))
            self.pre = precondition
            self.post = postcondition
            self.n_node_features = n_node_features

            s = Solver()
            s.add(And([Not(postcondition)] + [And(array_ref[i][j] >= self.final_bound.lo[i][j], array_ref[i][j] <= self.final_bound.hi[i][j]) for j in range(self.n_node_features) for i in range(self.n_nodes)]))
            for old_candidate in self.past_candidates:
                s.add(Not(And([old_candidate[i][j] == self.array_ref[i][j] for j in range(self.n_node_features) for i in range(self.n_nodes)])))
            r = s.check()
            if r == unsat or r == unknown:
                print("Input is safe (non overlapping final interval)")
                return
            m = s.model()
            self.current_candidate = tuple(tuple(m.eval(self.array_ref[i][j]) for j in range(self.n_node_features)) for i in range(self.n_nodes))
            self.past_candidates.add(self.current_candidate)

            self.visit(expr)

            if self.abort:
                # print("Discarding candidate and trying again")
                self.abort = False
            else:
                print("Analysis ended with counterexample, checking if feasible")
                s = Solver()
                s.add(And([precondition] + [self.array_ref[i][j] == self.current_candidate[i][j] for j in range(self.n_node_features) for i in range(self.n_nodes)]))
                r = s.check()

                if r == unsat or r == unknown:
                    # print("Discarding candidate and trying again")
                    self.abort = False
                else:
                    m = s.model()
                    ctr = tuple(tuple(m.eval(self.array_ref[i][j]) for j in range(self.n_node_features)) for i in range(self.n_nodes))
                    print("Counterexample")
                    print(ctr)
                    return ctr


        # wp = self.post
        # formula = Implies(self.pre, wp)
        # print("Formula generated")
        # # print(formula)
        # s = Solver()
        # s.add(Not(formula))
        # r = s.check()
        # if r == unsat:
        #     print("proved")
        # elif r == unknown:
        #     print("failed to prove")
        #     print(s.model())
        # else:
        #     print("counterexample")
        #     # print(s.model())
        #     m = s.model()
        #
        #     # generate the counterexample
        #     ctr = [[m.eval(self.array_ref[i][j]) for j in range(self.n_node_features)] for i in range(self.n_nodes)]
        #     print(ctr)

    def label(self, tree):
        return str(tree.children[0])

    @v_args(inline=True)
    def atom_op(self, op):
        if self.abort: return
        op = self.visit(op)
        psi = self.psi[op]
        pre_op, pre_n_node_labels = psi(self.array_ref, self.n_nodes, self.n_node_features)
        self.post = substitute(self.post, [(self.array_ref[i][j], pre_op[i][j]) for j in range(self.n_node_features) for i in range(self.n_nodes)])
        old_n_node_features = self.n_node_features
        self.n_node_features = pre_n_node_labels

        self.conditions.append(self.post)
        s = Solver()
        s.add(And([pre_op[self.node][j] == self.current_candidate[self.node][j] for j in range(old_n_node_features)]))
        r = s.check()
        if r == unsat:
            self.abort = True
            print("unsat at ", op)
        elif r == unknown:
            self.abort = True
        else:
            m = s.model()
            ctr = tuple(tuple(m.eval(self.array_ref[i][j]) for j in range(self.n_node_features)) for i in range(self.n_nodes))
            self.current_candidate = ctr
        # print(op)




    @visit_children_decor
    def lhd(self, args):
        if self.abort: return
        _phi, _sigma = args
        phi = self.phi[_phi]
        sigma = self.sigma[_sigma]

        # Message
        messages = {n: [] for n in range(self.n_nodes)}  # list of lists of messages
        pre_n_node_labels = self.n_node_features
        for src, label, tgt in zip(self.index_sources, self.e, self.index_targets):
            message, pre_n_node_labels = phi(self.array_ref[src], label, self.array_ref[tgt], self.n_node_features)
            messages[tgt].append(message)
        old_n_node_features = self.n_node_features
        self.n_node_features = pre_n_node_labels

        # Aggregate
        embeddings = {node: sigma(msgs, self.array_ref[node], self.n_node_features)[0] for node, msgs in messages.items()}
        self.post = substitute(self.post, [(self.array_ref[i][j], embeddings[i][j]) for j in range(self.n_node_features) for i in range(self.n_nodes)])
        self.n_node_features = sigma(messages[0], 0, self.n_node_features)[1]

        self.conditions.append(self.post)
        s = Solver()
        s.add(And([embeddings[self.node][j] == self.current_candidate[self.node][j] for j in range(old_n_node_features)]))
        r = s.check()
        if r == unsat:
            self.abort = True
            print("unsat at lhd")
        elif r == unknown:
            self.abort = True
        else:
            m = s.model()
            ctr = tuple(tuple(m.eval(self.array_ref[i][j]) for j in range(self.n_node_features)) for i in range(self.n_nodes))
            self.current_candidate = ctr


    @v_args(inline=True)
    def sequential_composition(self, left, right):
        if self.abort: return
        # reverse visit
        self.visit(right)
        self.visit(left)