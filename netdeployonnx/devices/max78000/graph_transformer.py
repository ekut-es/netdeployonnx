#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import onnx

from netdeployonnx.devices.max78000.optimizer import (
    Augment_Conv_Kernelshape,
    Augment_Conv_WeightsBias,
    Conv2DTranspose,
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
            Conv2DTranspose(),
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
