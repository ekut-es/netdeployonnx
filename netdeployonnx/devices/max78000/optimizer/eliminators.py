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
from .graph import Graph, Node
from .optimizer import NodeTransformType, Optimizer, logger


class EliminateQuantization1(Optimizer):
    """
    Eliminate quantization nodes from the graph
        inspired by github.com/onnx/optimizer
    """

    QUANTIZE = "QuantizeLinear"
    DEQUANTIZE = "DequantizeLinear"

    def match(self, graph: Graph, node: Node) -> bool:
        return node.op_type.startswith(self.QUANTIZE) and any(
            self.target(node).op_type.startswith(self.DEQUANTIZE)
        )

    def run_transformation(self, graph: Graph, node: Node) -> NodeTransformType:
        raise NotImplementedError("EliminateQuantization1::run_transformation")


class EliminateBatchNorm(Optimizer):
    """
    Eliminate batchnorm nodes from the graph
        inspired by github.com/onnx/optimizer
    """

    BATCHNORM = "BatchNormalization"

    def match(self, node: Node) -> bool:
        return node.op_type.startswith(self.BATCHNORM)

    def run_transformation(self, node: Node) -> NodeTransformType:
        # batchnorm is basically gamma*x_norm+beta, which is g*x+b
        # BN:  is Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
        # gemm is Y = alpha * A’ * B’ + beta * C,
        #
        original_input = node.input
        assert len(original_input) == 5, f"len(original_input)={len(original_input)}"
        X, scale, B, mean, var = (original_input + [""] * 5)[:5]  # noqa: N806
        node.input = [X, scale, B]  # TODO: improve this transformation, works for now
        node.output = [node.output[0]]
        node.op_type = "Gemm"
        return NodeTransformType.MODIFY


class EliminateDanglingNodes(Optimizer):
    """
    Eliminate dangling nodes from the graph
        inspired by github.com/onnx/optimizer
    """

    def run_on_graph(self, graph: Graph):
        """
        Before running the optimizer, we need to update the main graph
        """
        graph.update_maingraph()
        return super().run_on_graph(graph)

    def match(self, node: Node) -> bool:
        # return true if the source path does not lead to "input"
        # we need to search with a dfs for output
        if node.graph.maingraph is not None:
            return node.name not in node.graph.maingraph
        return False  # we cannot match the input node, if our maingraph is empty

    def run_transformation(
        self, node: Node, do_delete_convs: bool = False
    ) -> NodeTransformType:
        # logger.debug(f"deleting dangling node {node.name}")
        assert not do_delete_convs and node.op_type != "Conv", (
            "Are we sure we want" " to delete Convs?"
        )
        # as an optimization, remove all the subgraph
        if node.deleted is False:
            # this node is still in the graph, so we need to remove it
            node.deleted = True
            # now remove all input and output nodes with this function too
            for input_node in self.source(node)():
                self.run_transformation(input_node)
            for output_node in self.target(node)():
                self.run_transformation(output_node)
            return NodeTransformType.MODIFY  # we deleted the node ourselves
        return NodeTransformType.NO_CHANGE


class EliminateSqueeze(Optimizer):
    """
    Eliminate squeeze nodes from the graph
        inspired by github.com/onnx/optimizer
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Squeeze")

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"eliminating squeeze node {node.name}")
        # we need to change the input and output of the node
        assert len(node.input) == 1
        assert len(node.output) == 1
        # now we need to find the source node
        source_nodes = self.source(node)()
        assert (
            len(source_nodes) == 1
        ), f"source_nodes={source_nodes}; node.input={node.input}"
        source_node = source_nodes[0]

        # now we need to find the target node
        target_nodes = self.target(node)()
        assert (
            len(target_nodes) == 1
        ), f"target_nodes={target_nodes}; node.output={node.output}"
        target_node = target_nodes[0]

        # now we need to replace the input of the target node
        target_node.input = [source_node.output[0]]

        # now we need to destroy the node
        return NodeTransformType.DESTROY


class EliminatePassthrough(Optimizer):
    """
    Eliminate passthrough nodes from the graph
        inspired by github.com/onnx/optimizer
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Pass")

    @classmethod
    def replace_input(cls, node: Node, old_input: str, new_input: str):
        inputs = [input for input in node.input]
        input_idx = inputs.index(old_input)
        inputs.remove(old_input)
        inputs.insert(input_idx, new_input)
        node.input = inputs

    @classmethod
    def replace_output(cls, node: Node, old_output: str, new_output: str):
        outputs = [output for output in node.output]
        output_idx = outputs.index(old_output)
        outputs.remove(old_output)
        outputs.insert(output_idx, new_output)
        node.output = outputs

    def handle_input(self, node: Node, new_connection_name: str):
        # first, find all inputs
        assert len(node.input) == 1
        # we need to change only input or output, depending on the case
        if node.input[0] in node.graph.input:
            # we are the first node, so we can just return
            return
        # now we need to find the source node
        source_nodes = self.source(node)()
        assert (
            len(source_nodes) == 1
        ), f"source_nodes={source_nodes}; node.input={node.input}"
        source_node = source_nodes[0]

        # now we need to replace the input of the source node
        self.replace_output(source_node, node.input[0], new_connection_name)

    def handle_output(self, node: Node, new_connection_name: str):
        # now, assure that we only have 1 output
        assert len(node.output) == 1
        # logger.debug(node.output[0], graph.output)
        if node.output[0] in node.graph.output:
            # we are the last node, so we can just return
            return
        # now we need to find the target node
        target_nodes = self.target(node)()
        assert (
            len(target_nodes) == 1
        ), f"target_nodes={target_nodes}; node.output={node.output}"
        target_node = target_nodes[0]

        # now we need to replace the output of the target node
        self.replace_input(target_node, node.output[0], new_connection_name)

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"rerouting output of node {node.name}")
        new_connection_name = f"pass_{node.name}"
        if node.output[0] in node.graph.output:
            new_connection_name = node.output[0]
        if node.input[0] in node.graph.input:
            new_connection_name = node.input[0]

        self.handle_input(node, new_connection_name)
        self.handle_output(node, new_connection_name)

        # now we need to destroy the node
        return NodeTransformType.DESTROY
