def find_safe_ptb(model, dataset, domain, node):
    original_y = dataset[0].y
    graph = MGExplainer(model).explain(node, next(iter(SingleGraphLoader(dataset).load()))[0], None, False)
    graph.y = original_y
    n_nodes = graph.n_nodes
    node_list = sorted(list(set(graph.a.row.tolist())))
    mapping = lambda xx: node_list.index(xx)
    rev_mapping = lambda xx: node_list[xx]

    a_tensor = tf.sparse.SparseTensor(
        tf.stack(
            [tf.convert_to_tensor(list(map(mapping, graph.a.row)), dtype=tf.int64), tf.convert_to_tensor(list(map(mapping, graph.a.col)), dtype=tf.int64)],
            axis=1),
        tf.convert_to_tensor(graph.a.data),
        tf.convert_to_tensor((n_nodes, n_nodes), dtype=tf.int64)
    )
    rev_a = tf.sparse.SparseTensor(
        tf.stack([tf.convert_to_tensor(graph.a.row, dtype=tf.int64), tf.convert_to_tensor(graph.a.col, dtype=tf.int64)], axis=1),
        tf.convert_to_tensor(graph.a.data),
        tf.convert_to_tensor(graph.a.shape, dtype=tf.int64)
    )
    a = domain.abstract_adj(a_tensor)
    rev_a = domain.abstract_adj(rev_a)
    y = tf.gather(tf.convert_to_tensor(graph.y), node_list)

    abs_psi_dict = {'lin': domain.make_layer(model.trainable_variables[0].value.numpy(), 'RELU'),
                    'out': domain.make_layer(model.trainable_variables[1].value.numpy(), 'linear')}

    x_tensor = tf.convert_to_tensor(graph.x)
    x_torch_tensor = torch.tensor(x_tensor.numpy())

    e_tensor = tf.convert_to_tensor(graph.e)
    e = domain.abstract(e_tensor)
    # e = domain.abstract(e_tensor)

    interp = AbstractInterpreter(domain,
                                 abs_psi_dict,
                                 {'x': domain.prod}, {'+': domain.sm})

    mapped_node = mapping(node)
    pred = model((x_tensor, a_tensor, e_tensor))


    predicted_class = concrete_prediction(pred, mapped_node)
    print('looking for negative bound')
    untested_delta = 0.1
    safe_delta = 0
    unsafe_delta = None
    # last_success = True
    while True:
        x = domain.abstract(x_tensor, x_L=x_torch_tensor - untested_delta)
        lb, ub = domain.run_abstract_model(interp.run(x, a, e, model.expr))
        abs_pred = abstract_prediction(lb, ub, mapped_node)
        if abs_pred == predicted_class:  # untested delta is safe
            if abs(safe_delta - untested_delta) <= 0.001:
                break
            safe_delta = untested_delta
            untested_delta = safe_delta * 2 if unsafe_delta is None else (unsafe_delta + safe_delta) / 2  # increase untested delta
            print('safe delta: ', safe_delta)
            print('next test', untested_delta)
            # if last_success:
            #     delta, prev_delta = delta * 2, delta
            # else:
            #     delta, prev_delta = abs(delta + prev_delta) / 2, delta
            # last_success = True
        else: # crossed the decision boundary, find untested delta between current untested and safe delta
            unsafe_delta = untested_delta
            untested_delta = (unsafe_delta + safe_delta) / 2
            print('unsafe delta: ', unsafe_delta)
            print('next test', untested_delta)
            # if last_success:
            #     delta, prev_delta = abs(delta + prev_delta) / 2, delta
            # else:
            #     delta, prev_delta = delta / 2, delta
            # last_success = False
    print('negative bound:', untested_delta)
    neg_bound = untested_delta


    print('looking for positive bound')
    untested_delta = 0.1
    safe_delta = 0
    unsafe_delta = None
    # last_success = True
    while True:
        x = domain.abstract(x_tensor, x_U=x_torch_tensor + untested_delta)
        lb, ub = domain.run_abstract_model(interp.run(x, a, e, model.expr))
        abs_pred = abstract_prediction(lb, ub, mapped_node)
        if abs_pred == predicted_class:  # untested delta is safe
            if abs(safe_delta - untested_delta) <= 0.001:
                break
            safe_delta = untested_delta
            untested_delta = safe_delta * 2 if unsafe_delta is None else (unsafe_delta + safe_delta) / 2  # increase untested delta
            print('safe delta: ', safe_delta)
            print('next test', untested_delta)
            # if last_success:
            #     delta, prev_delta = delta * 2, delta
            # else:
            #     delta, prev_delta = abs(delta + prev_delta) / 2, delta
            # last_success = True
        else:  # crossed the decision boundary, find untested delta between current untested and safe delta
            unsafe_delta = untested_delta
            untested_delta = (unsafe_delta + safe_delta) / 2
            print('unsafe delta: ', unsafe_delta)
            print('next test', untested_delta)
            # if last_success:
            #     delta, prev_delta = abs(delta + prev_delta) / 2, delta
            # else:
            #     delta, prev_delta = delta / 2, delta
            # last_success = False
    print('positive bound:', untested_delta)
    pos_bound = untested_delta

    print('sanity check')
    print('testing with negative bound', neg_bound, ' positive bound ', pos_bound)
    x = domain.abstract(x_tensor, x_L=x_torch_tensor-neg_bound)
    lb, ub = domain.run_abstract_model(interp.run(x, a, e, model.expr))
    abs_pred = abstract_prediction(lb, ub, mapped_node)
    print(abs_pred)
    print(abs_pred == predicted_class)
    print_bounds(lb[:, mapped_node:mapped_node + 1, :], ub[:, mapped_node:mapped_node + 1, :], (pred[0][mapped_node:mapped_node + 1],),
                 y[mapped_node:mapped_node + 1])


    x = domain.abstract(x_tensor, x_U=x_torch_tensor + pos_bound)
    lb, ub = domain.run_abstract_model(interp.run(x, a, e, model.expr))
    abs_pred = abstract_prediction(lb, ub, mapped_node)
    print(abs_pred)
    print(abs_pred == predicted_class)
    print_bounds(lb[:, mapped_node:mapped_node + 1, :], ub[:, mapped_node:mapped_node + 1, :], (pred[0][mapped_node:mapped_node + 1],),
                 y[mapped_node:mapped_node + 1])

    x = domain.abstract(x_tensor, x_L=x_torch_tensor-neg_bound, x_U=x_torch_tensor + pos_bound)
    lb, ub = domain.run_abstract_model(interp.run(x, a, e, model.expr))
    abs_pred = abstract_prediction(lb, ub, mapped_node)
    print(abs_pred)
    print(abs_pred == predicted_class)
    print_bounds(lb[:, mapped_node:mapped_node + 1, :], ub[:, mapped_node:mapped_node + 1, :], (pred[0][mapped_node:mapped_node + 1],),
                 y[mapped_node:mapped_node + 1])
    # check_soundness(pred, lb, ub)
