from .graph import Node
from .optimizer import NodeTransformType, Optimizer


class Augment_Conv_WeightsBias(Optimizer):  # noqa: N801
    """
    Augment Conv nodes with additional information
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Conv") and len(node.input) != 3

    def run_transformation(self, node: Node) -> NodeTransformType:
        if len(node.input) == 2:
            # add bias
            node.input.append(node.name + "_bias")
            return NodeTransformType.MODIFY
        elif len(node.input) == 1:
            # add bias and weight
            node.input.append(node.name + "_weight")
            node.input.append(node.name + "_bias")
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
