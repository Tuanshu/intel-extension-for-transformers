
def ts_compile(model, config=None):
    """The compile interface.

    Firstly, use model loader to get the computation graph with corresponding framework.
    The graph contains nodes and edges, the node is op and the edge is the tensor.
    Then extract the ops in the graph and pack them to our form.
    Next exploit these above ops to consist sub-graph, which can see as "a new big op", like LayerNorm.

    Note:
        There may have different computation flow in one subgraph.
    Finally, convert them to .yaml file and .bin file for model configuration and inference.
    """
    from intel_extension_for_transformers.llm.runtime.deprecated.compile.compile import Optimizer,_config_validation,start_pipeline,_dynamic_quantization
    from intel_extension_for_transformers.llm.runtime.deprecated.compile.graph_utils import get_autocast_info,autocast_init

    try:
        get_autocast_info()
    except:
        autocast_init()

    config = _config_validation(config)
    model = start_pipeline(model, config=config)

    optimizer = Optimizer(model)
    optimizer.optimize()
    if get_autocast_info()['cast_type'] == "dynamic_int8":
        model = _dynamic_quantization(model)

    return model