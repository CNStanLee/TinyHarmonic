class MoveLinearPastEltwiseMul(Transformation):
    """Move linear operations (mul) past elementwise mul operations where possible.
    Specifically,matches and transforms the following patterns:
    (x*A) * (y*B) -> (xy)*(A*B)
    where x and y are dynamic inputs, A, B are constant tensors (in general).
    """

    def move_node(self, graph, n, prod0, prod1, node_ind):
        # found! move one of the muls to output, remove the other one
        lin0_in0 = prod0.input[0]
        lin1_in0 = prod1.input[0]
        in0 = n.input[0]
        out = n.output[0]
        # TODO: check shapes don't change through scalar mul or add
        # connect the eltwise add inputs to mul inputs
        n.input[0] = lin0_in0
        n.input[1] = lin1_in0
        # connect mul0 output to eltwise add output
        prod0.output[0] = out
        # connect the input of mul0 and output of eltwise add together
        n.output[0] = in0
        prod0.input[0] = in0
        # move prod0 node past eltwise add node, and remove prod1
        graph.node.remove(prod1)
        graph.node.remove(prod0)
        graph.node.insert(node_ind - 2, prod0)

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Mul": #Checking if node is Eltwisemul.
                # scalar add has an initializer on one input
                in0 = n.input[0]
                in1 = n.input[1]
                if in0 is None or in1 is None:
                    continue
                A = model.get_initializer(in0)
                B = model.get_initializer(in1)
                if A is not None or B is not None:
                    continue
                # check for mul with same initializer on both inputs
                prod0 = model.find_producer(in0)
                prod1 = model.find_producer(in1)
                # Also check case when both branches are empty and come
                # from the same node: (prod0 == prod1)
                # Other transform should handle that
                if prod0 is None or prod1 is None or (prod0 == prod1):
                    continue
                if len(prod0.input) < 2 or len(prod1.input) < 2:
                    continue
                init0 = model.get_initializer(prod0.input[1])
                init1 = model.get_initializer(prod1.input[1])
                # if either initializer is None, skip
                if init0 is None or init1 is None:
                    continue
                if prod0.op_type == "Mul" and prod1.op_type == "Mul": #  (x*A) * (y*B) -> (xy)*(AB)
                    # Adding the update intializer condition in case of EltwiseMul.
                    init = init0*init1                         #Updating the initializer of the node which will move past the EltwiseMul
                    model.set_initializer(prod0.input[1],init) # update initializer of prod0, the node which will move.
                    self.move_node(graph,n,prod0,prod1,node_ind)
                    node_ind -= 1
                    graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)