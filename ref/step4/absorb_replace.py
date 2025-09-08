class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            # search for (MultiThreshold, Add) pair
            node_ind += 1
            if (
                n.op_type == "MultiThreshold"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Add":
                    mt_node = n
                    add_node = consumer
                    threshold_name = mt_node.input[1]
                    add_weight_name = add_node.input[1]
                    print("Threshold name : ",threshold_name)
                    print("Add Weight name : ",add_weight_name)
                    T = model.get_initializer(threshold_name)
                    A = model.get_initializer(add_weight_name)
                    # mt_inst_unsigned = getCustomOp(mt_node)
                    # usigned_dtype = mt_inst_unsigned.get_nodeattr("out_dtype")
                    # if(usigned_dtype == 'UINT8'): #Assuming that with UINT8 inputs the bias factor needs to be 0 for the MT node.
                    #     A=A
                    if (A is None) or (T is None):
                        warnings.warn("Threshold or add bias not constant, skipping")
                        continue
                    end_name = add_node.output[0]
                    # we can only absorb scalar adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    if not is_scalar:
                        continue
                    bias = A.flatten()[0]
                    # set MultiThreshold bias property
                    mt_inst = getCustomOp(mt_node)
                    bias += mt_inst.get_nodeattr("out_bias")
                    mt_inst.set_nodeattr("out_bias", bias)
                    graph_modified = True
                    # compute new DataType for MultiThreshold output
                    steps = T.shape[-1]
                    new_min = bias
                    new_max = steps + bias
                    odt = DataType.get_smallest_possible(steps).name.replace("UINT", "INT")
                    print(odt)
                    print("-----------------------------------")
                    odt = DataType[odt]
                    # assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    # ), """Could
                    # not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #     new_min,
                    #     new_max,
                    # )
                    # mt_inst.set_nodeattr("out_dtype", odt.name)
                    # print(odt)
                    # print(odt)
                    #-----------------------
                    # if(usigned_dtype != 'UINT8'):
                    #     assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    #     ), """Could
                    #     not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #         new_min,
                    #         new_max,
                    #     )
                    #     mt_inst.set_nodeattr("out_dtype", odt.name)
                    # if(usigned_dtype == 'UINT8'):
                    #     odt = 'UINT8'
                    #     usigned_dtype = ' '
                    #     mt_inst.set_nodeattr("out_dtype", 'UINT8')
                    #-------------------------
                    # print(odt.name)
                    #--------------------------------------------------------
                    if(threshold_name != "MultiThreshold_8_param0" and threshold_name != "MultiThreshold_15_param0"):
                        assert odt.allowed(new_max) and odt.allowed(
                        new_min
                        ), """Could
                        not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                        new_min,
                        new_max,
                        )
                        mt_inst.set_nodeattr("out_dtype", odt.name)
                    if(threshold_name == "MultiThreshold_8_param0" or threshold_name == "MultiThreshold_15_param0"):
                        mt_inst.set_nodeattr("out_dtype", "INT7")
                    # remove Add node, rewire MultiThreshold
                    graph.node.remove(add_node)
                    mt_node.output[0] = end_name
                    # set datatype
                    model.set_tensor_datatype(end_name, odt)
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
