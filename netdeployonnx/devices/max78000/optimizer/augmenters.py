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
import numpy
import onnx

from .graph import Graph, Node
from .optimizer import NodeTransformType, Optimizer


class Augment_Conv_WeightsBias(Optimizer):  # noqa: N801
    """
    Augment Conv nodes with additional information
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Conv") and len(node.input) != 3

    def run_transformation(self, node: Node) -> NodeTransformType:
        weight_name = node.name + "_weight"
        bias_name = node.name + "_bias"
        # check if there is kernel shape or an already existing weights

        if len(node.input) >= 2:
            input_value = Graph.resolve_input_name(node.graph.onnx(), node.input[1])
            if input_value is not None:
                weight_shape = input_value.shape
            else:
                raise NotImplementedError(
                    f"conv-weight not found (node.name={node.name})"
                )
        else:
            raise NotImplementedError("no weight on the conv?!")
            weight_shape = (1, 1, 1, 1)
            weight = onnx.numpy_helper.from_array(
                arr=numpy.zeros(shape=weight_shape), name=weight_name
            )
        bias = onnx.numpy_helper.from_array(
            arr=numpy.zeros(shape=(weight_shape[0])),  # bias is applied to each output
            name=bias_name,
        )
        if len(node.input) == 2:
            # add bias
            node.input.append(bias_name)
            # we cant just add the bias, we would have to add a node / initializer too
            # we prefer bias to be an initializer

            node.graph.initializers[bias_name] = (
                bias.data_type,
                bias.dims,
                bias.raw_data,
            )
            return NodeTransformType.MODIFY
        elif len(node.input) == 1:
            # add bias and weight
            raise Exception("dude no.")
            node.input.append(weight_name)
            node.input.append(bias_name)
            node.graph.initializers[bias_name] = (
                bias.data_type,
                bias.dims,
                bias.raw_data,
            )
            node.graph.initializers[weight_name] = (
                weight.data_type,
                weight.dims,
                weight.raw_data,
            )
            return NodeTransformType.MODIFY
        return NodeTransformType.NO_CHANGE


class Augment_Conv_Kernelshape(Optimizer):  # noqa: N801
    """
    Augment Conv nodes with additional information
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Conv") and "kernel_shape" not in node.attributes

    def run_transformation(self, node: Node) -> NodeTransformType:
        # we need to find out the shape of the kernel
        # we can do this by looking at the weight tensor
        # and then infer the shape from that
        kernel_shape = [1, 1]
        assert len(node.input) >= 2
        weight_node = node.input[1]
        if weight_node in node.graph.input:
            input_val = node.graph.input[weight_node]
            kernel_shape = [dim.dim_value for dim in input_val.tensor_type.shape.dim]
            kernel_shape = kernel_shape[-2:]  # fetch the last two dimensions
        else:
            source_nodes = self.source(node)()
            for source_node in source_nodes:
                if weight_node in source_node.output:
                    kernel_shape = source_node.attributes["kernel_shape"]
                    break
            else:
                raise Exception("Weight node not found")

        assert kernel_shape in [[1, 1], [3, 3]], "kernel shape must be 1x1 or 3x3"

        node.attributes["kernel_shape"] = kernel_shape
        return NodeTransformType.MODIFY


class Conv2DTranspose(Optimizer):
    """
    Augment ConvTranspose nodes with additional information
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("ConvTranspose")

    def run_transformation(self, node: Node) -> NodeTransformType:
        # we need to find out the shape of the kernel
        # we can do this by looking at the weight tensor
        # and then infer the shape from that
        node.op_type = "Conv"
        node.attributes["transpose"] = 1
        return NodeTransformType.MODIFY
