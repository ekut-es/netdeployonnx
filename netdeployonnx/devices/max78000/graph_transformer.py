import onnx

from netdeployonnx.devices.max78000.optimizer import (
    Augment_Conv_Kernelshape,
    Augment_Conv_WeightsBias,
    EliminateDanglingNodes,
    EliminatePassthrough,
    FuseAddClip,
    FuseBatchNorm,
    FuseClipQuantization,
    FuseConvAvgPool,
    # FuseSqueeze,
    FuseConvDiv,
    FuseConvFactorSqueeze,
    FuseConvMaxPool,
    FuseConvSqueeze,
    FuseFlatten,
    FuseGemmConvRelu,
    FuseMulPowFactor,
    FusePoolClip,
    FuseQuantizeDequantizeLinear,
    FuseReshape,
    Graph,
    ReplaceMatMulWithGemm,
    logger,
)


def run_optimizer(graph: Graph, last_pass=False) -> int:
    # now run
    optimizers = (
        [
            Augment_Conv_WeightsBias(),
            ReplaceMatMulWithGemm(),
            # FuseSqueeze(),
            FuseQuantizeDequantizeLinear(),
            FuseClipQuantization(),
            EliminatePassthrough(),
            FuseBatchNorm(),
            FuseMulPowFactor(),
            FuseConvDiv(),
            FuseConvSqueeze(),
            FuseGemmConvRelu(),
            FuseConvFactorSqueeze(),
            FuseConvMaxPool(),
            FuseConvAvgPool(),
            FuseFlatten(),
            FuseAddClip(),
            FusePoolClip(),
            FuseReshape(),
            Augment_Conv_Kernelshape(),
        ]
        if not last_pass
        else [
            EliminateDanglingNodes(),
        ]
    )
    num_changes = 0

    for optimizer in optimizers:
        logger.debug(f"Running optimizer {optimizer.__class__.__name__}")
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
