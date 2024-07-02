import onnx

from netdeployonnx.devices.max78000.optimizer import (
    EliminateDanglingNodes,
    EliminatePassthrough,
    FuseClipQuantization,
    FuseConvMaxPool,
    FuseConvRelu,
    FuseConvReshape,
    FuseConvSqueeze,
    FuseSqueeze,
    Graph,
    logger,
)


def run_optimizer(graph: Graph, last_pass=False) -> int:
    # now run
    optimizers = (
        [
            FuseSqueeze(),
            FuseClipQuantization(),
            EliminatePassthrough(),
            FuseConvSqueeze(),
            FuseConvRelu(),
            FuseConvMaxPool(),
            FuseConvReshape(),
        ]
        if not last_pass
        else [
            EliminateDanglingNodes(),
        ]
    )
    num_changes = 0

    for optimizer in optimizers:
        logger.info(f"Running optimizer {optimizer.__class__.__name__}")
        num_changes += optimizer.run_on_graph(graph)

    # iterate over all nodes in the graph
    return num_changes


def transform_graph(graph: onnx.GraphProto) -> onnx.GraphProto:
    # run optimizer as long as the replacement_list stays empty
    # copy graph
    # debugpy.breakpoint()

    transform_graph = Graph(graph)
    last_pass = False
    while True:
        changes: int = run_optimizer(transform_graph, last_pass=last_pass)
        if changes == 0:
            if last_pass:
                break
            else:
                last_pass = True

    return transform_graph.onnx()
