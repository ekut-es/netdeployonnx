from .graph import Node
from .optimizer import NodeTransformType, Optimizer, logger


class FuseAnyGenericTarget(Optimizer):
    def __init__(
        self,
        node_optypes: list[str],
        node_optyes_inputsizes: list[int],
        target_optype: str,
        target_optype_outputcount: int = 1,
    ):
        self.node_optypes = node_optypes
        self.node_optyes_inputsizes = node_optyes_inputsizes
        self.target_optype = target_optype
        self.target_optype_outputcount = target_optype_outputcount

    @property
    def optyps(self) -> str:
        return "/".join(self.node_optypes)

    @property
    def node_optyes_inputsizes_str(self) -> str:
        return " or ".join(str(x) for x in self.node_optyes_inputsizes)

    def match(self, node: Node) -> bool:
        # only match if all the outputs are target_optype
        matches_this = any(
            node.op_type.startswith(node_optype) for node_optype in self.node_optypes
        )
        matches_target = all(self.target(node).op_type.startswith(self.target_optype))
        return matches_this and matches_target

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a a generic layer, that is followed by a other layer
        """
        assert len(node.input) in self.node_optyes_inputsizes, (
            f"{self.optyps} should have {self.node_optyes_inputsizes_str} inputs"
            f", but has {len(node.input)}"
        )
        assert len(node.output) == 1, f"{self.optyps} should have only one output"

        # delete the following generic
        outputs = set()
        for target_node in self.target(node)():
            if target_node.deleted:
                continue
            if not target_node.op_type.startswith(self.target_optype):
                # we should not be able to fuse that then?
                raise ValueError(
                    f"We can only fuse if all outputs are {self.target_optype}"
                )
            assert len(target_node.output) == self.target_optype_outputcount, (
                f"{self.target_optype} should have "
                f"{self.target_optype_outputcount} output(s)"
                f", but has {len(target_node.output)} [{target_node.output}]"
            )
            outputs |= set([target_node.output[0]])
            for attrname, attrval in target_node.attributes.items():
                newattr = f"_{self.target_optype.lower()}_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            target_node.deleted = True
        if list(outputs) == 0:
            raise ValueError(f"We should have found a {self.target_optype} node")
        # now we need to replace the output of the node
        assert len(node.output) <= len(outputs), "we should not loose outputs?!"
        node.output = list(outputs)
        node.name = "/".join(node.name.split("/")[:] + ["_"] + [self.target_optype])
        assert len(node.output) == 1, f"outputs={node.output}"
        return NodeTransformType.MODIFY


class FuseAnyGenericSource(Optimizer):
    def __init__(
        self,
        node_optypes: list[str],
        node_optyes_inputsizes: list[int],
        source_optype: str,
    ):
        self.node_optypes = node_optypes
        self.node_optyes_inputsizes = node_optyes_inputsizes
        self.source_optype = source_optype

    @property
    def optyps(self) -> str:
        return "/".join(self.node_optypes)

    @property
    def node_optyes_inputsizes_str(self) -> str:
        return " or ".join(str(x) for x in self.node_optyes_inputsizes)

    def match(self, node: Node) -> bool:
        # only match if all the outputs are source_optype
        matches_this = any(
            node.op_type.startswith(node_optype) for node_optype in self.node_optypes
        )
        matches_source = any(self.source(node).op_type.startswith(self.source_optype))
        return matches_this and matches_source

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a Gemm layer, that is followed by a relu layer
        """
        assert (
            len(node.input) in self.node_optyes_inputsizes
        ), f"{self.optyps} should have {self.node_optyes_inputsizes_str} inputs:"
        assert len(node.output) == 1, "Conv/Gemm should have only one output"

        # delete the following generic
        inputs: list[str] = []
        for source_node in self.source(node)():
            if source_node.deleted:
                inputs.append(None)
                continue
            if not source_node.op_type.startswith(self.source_optype):
                # we should not be able to fuse that then?
                inputs.append(None)
                continue
            assert (
                len(source_node.input) >= 1
            ), f"{self.source_optype} should have atleast 1 input"

            # we assume our optype only has one input
            inputs.append(source_node.input[0])

            for attrname, attrval in source_node.attributes.items():
                newattr = f"_{self.source_optype.lower()}_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(
                        f"Attribute {newattr} already exists in node" f" {node.name}"
                    )
                node.attributes[newattr] = attrval
            source_node.deleted = True
            # we cant break, as for example GEMM have multiple inputs
        if inputs:
            node.input = [
                inputs[orig_i]
                if len(inputs) > orig_i and inputs[orig_i]
                else orig_input
                for orig_i, orig_input in enumerate(node.input)
            ]
        else:
            raise ValueError(f"We should have found a {self.source_optype} node")

        node.name = "/".join(node.name.split("/")[:] + ["_"] + [self.source_optype])
        assert len(node.output) == 1, f"input={node.input}"
        return NodeTransformType.MODIFY


class FuseGemmConvGenericTarget(FuseAnyGenericTarget):
    def __init__(
        self,
        target_optype: str,
        target_optype_outputcount: int = 1,
    ):
        super().__init__(
            node_optypes=["Gemm", "Conv"],
            node_optyes_inputsizes=[2, 3],  # X, W, B
            target_optype=target_optype,
            target_optype_outputcount=target_optype_outputcount,
        )


class FuseGemmConvGenericSource(FuseAnyGenericSource):
    def __init__(self, source_optype: str):
        super().__init__(
            node_optypes=["Gemm", "Conv"],
            node_optyes_inputsizes=[2, 3],  # X, W, B
            source_optype=source_optype,
        )


class FuseGemmConvRelu(FuseGemmConvGenericTarget):
    """
    Fuse conv+relu nodes from the graph
    """

    def __init__(self):
        super().__init__("Relu")

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a conv layer, that is followed by a relu layer
        """
        ret = super().run_transformation(node)
        node.attributes["activation"] = "relu"
        return ret


class FuseBatchNorm(FuseGemmConvGenericTarget):
    """
    Fuse batchnorm up to conv/gemm
    """

    def __init__(self):
        super().__init__("BatchNormalization", 3)  # batch norm has 3 outputs


class FuseConvMaxPool(FuseGemmConvGenericSource):
    """
    Fuse conv+maxpool nodes from the graph
    """

    def __init__(self):
        super().__init__("MaxPool")


class FuseConvAvgPool(FuseGemmConvGenericSource):
    """
    Fuse conv+avgpool nodes from the graph
    """

    def __init__(self):
        super().__init__("AveragePool")


class FuseReshape(FuseGemmConvGenericSource):
    def __init__(self):
        super().__init__("Reshape")


class FuseFlatten(FuseGemmConvGenericSource):
    def __init__(self):
        super().__init__("Flatten")


class FuseConvFactorSqueeze(FuseGemmConvGenericTarget):
    def __init__(self):
        super().__init__("Factor")


class FuseConvDiv(FuseGemmConvGenericTarget):
    def __init__(self):
        super().__init__("Div")


class FuseQuantizeDequantizeLinear(Optimizer):
    """
    Fuse quantize and dequantize linear nodes from the graph to a pass node
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("DequantizeLinear") and any(
            self.source(node).op_type.startswith("QuantizeLinear")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        modifiy the dequant node (this one), so it is a pass
        then reroute the input of the quantize node
        then delete the quantize node
        """
        logger.debug(f"running transformation on node {node.op_type} {node.name}")

        # we need to find the quantize node
        source_nodes = self.source(node)()
        assert len(source_nodes) >= 1, f"len={len(source_nodes)}"
        quantize_node = source_nodes[0]
        assert quantize_node
        assert quantize_node.op_type.startswith("QuantizeLinear")

        # we are targeting the dequantize node
        node.op_type = "Pass"
        node.name = "/".join(node.name.split("/")[:] + ["_"] + ["Pass"])

        # now set our nodes input to the previous nodes output
        node.attributes["dequantize_x_scale"] = node.input[1]
        node.attributes["dequantize_x_zero_point"] = node.input[2]
        node.attributes["quantize_y_scale"] = quantize_node.input[1]
        node.attributes["quantize_y_zero_point"] = quantize_node.input[2]
        node.input = [quantize_node.input[0]]
        # now we need to destroy the quantize node
        quantize_node.deleted = True

        return NodeTransformType.MODIFY


class FuseClipQuantization(FuseGemmConvGenericTarget):
    def __init__(self):
        super().__init__("Clip")


class FuseAddClip(FuseAnyGenericTarget):
    def __init__(self):
        super().__init__(
            node_optypes=["Add"],
            node_optyes_inputsizes=[2],  # Add has usually 2
            target_optype="Clip",
        )


class FusePoolClip(FuseAnyGenericTarget):
    """not sure if this is in the best ideas (is it collision free with other rules?)"""

    def __init__(self):
        super().__init__(
            node_optypes=["AveragePool", "MaxPool"],
            node_optyes_inputsizes=[1],  # pool has usually 1
            target_optype="Clip",
        )


# class FuseClipQuantization(Optimizer):
#     """
#     Fuse clip quantization from the graph
#     """

#     def match(self, node: Node) -> bool:
#         return node.op_type.startswith("Clip") and any(
#             self.target(node).op_type.startswith("Clip")
#         )

#     def run_transformation(self, node: Node) -> NodeTransformType:
#         logger.debug(f"running transformation on node {node.op_type} {node.name}")
#         # we are targeting the first clip
#         # so we have one output guaranteed
#         assert len(node.output) == 1
#         # and we are guaranteed to have two or three inputs
#         assert len(node.input) in [2, 3]
#         # we need to find the second clip node
#         target_nodes = self.target(node)()
#         second_clip_node = next(
#             node for node in target_nodes if node.op_type.startswith("Clip")
#         )

#         previous_nodes = self.source(node)()
#         previous_outputs: set[str] = {
#             output for node in previous_nodes for output in node.output
#         }
#         # we need to find which is the good input, which is one of the outputs of our
#         # previous node
#         good_input = next(input for input in node.input if input in previous_outputs)
#         assert good_input, f"good_input={good_input}"
#         good_output = list(second_clip_node.output)[0]
#         assert good_output, f"good_output={good_output}"

#         # now we change this node
#         node.op_type = "Pass"
#         node.name = "/".join(node.name.split("/")[:] + ["_"] + ["Pass"])
#         node.input = [good_input]
#         node.output = [good_output]

#         # we need to destroy the second clip node
#         second_clip_node.deleted = True
#         return NodeTransformType.MODIFY


class FuseSqueeze(Optimizer):
    """
    Fuse Mul(Div(), Pow()) quantization nodes from the graph
    """

    DIV = "Div"
    MUL = "Mul"
    POW = "Pow"

    def match(self, node: Node) -> bool:
        return (
            node.op_type.startswith(self.MUL)
            and any(self.source(node).op_type.startswith(self.DIV))
            and any(self.source(node).op_type.startswith(self.POW))
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"FuseSqueeze on node {node.op_type} {node.name}")
        # we are targeting MUL
        # so we have one output guaranteed
        assert len(node.output) == 1
        # and we are guaranteed to have two inputs
        assert len(node.input) == 2
        # we need to find the div node
        source_nodes = self.source(node)()
        assert len(source_nodes) == 2, f"len={len(source_nodes)}"  # div and pow
        div_node = next(
            node for node in source_nodes if node.op_type.startswith(self.DIV)
        )
        # we need to find the pow node
        pow_node = next(
            node for node in source_nodes if node.op_type.startswith(self.POW)
        )

        # we need to destroy the div and pow nodes
        div_node.deleted = True
        pow_node.deleted = True

        div_value = node.graph.get_const_value(div_node.input[1])

        pow_base = node.graph.get_const_value(pow_node.input[0])
        pow_expo = node.graph.get_const_value(pow_node.input[1])

        # we need to modify this node
        node.op_type = "Squeeze"
        node.name = "/".join(node.name.split("/")[:]) + "_" + "/Squeeze"
        # factor wieder drinne
        factor = pow_base**pow_expo / div_value
        node.attributes["factor"] = [factor]
        # TODO: rename outputs?
        node.input = [list(div_node.input)[0]]

        return NodeTransformType.MODIFY


class FuseMulPowFactor(Optimizer):
    MUL = "Mul"
    POW = "Pow"

    def match(self, node: Node) -> bool:
        return node.op_type.startswith(self.MUL) and any(
            self.source(node).op_type.startswith(self.POW)
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"FuseMulPowFactor on node {node.op_type} {node.name}")
        # we are targeting MUL
        # so we have one output guaranteed
        assert len(node.output) == 1
        # and we are guaranteed to have two inputs
        assert len(node.input) == 2
        # we need to find the div node
        source_nodes = self.source(node)()
        assert len(source_nodes) == 2, f"len={len(source_nodes)}"  # div and pow
        # we need to find the pow node
        pow_node = next(
            node for node in source_nodes if node.op_type.startswith(self.POW)
        )
        pow_node_output_name = pow_node.output[0]
        assert len(pow_node_output_name) > 0

        # we need to destroy the div and pow nodes
        pow_node.deleted = True

        pow_base = node.graph.get_const_value(pow_node.input[0])
        pow_expo = node.graph.get_const_value(pow_node.input[1])

        # we need to modify this node
        node.op_type = "Factor"
        node.name = "/".join(node.name.split("/")[:]) + "_" + "/Factor"
        # factor wieder drinne
        factor = pow_base**pow_expo
        node.attributes["factor"] = [factor]
        # TODO: rename outputs?
        node.input = [i for i in node.input if i != pow_node_output_name]

        return NodeTransformType.MODIFY


class FuseConvSqueeze(Optimizer):
    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Squeeze") and any(
            self.source(node).op_type.startswith("Conv")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"FuseSqueeze on node {node.op_type} {node.name}")
        # we are targeting Squeeze
        source_nodes = self.source(node)()
        assert len(source_nodes) == 1, f"len={len(source_nodes)}"
        source_node = source_nodes[0]
        assert source_node
        assert source_node.op_type.startswith("Conv")
        # we need to modify this node
        for attrname, attrval in node.attributes.items():
            source_node.attributes[f"_squeeze_{attrname}"] = attrval

        # now fix the connections
        target_nodes = self.target(node)()
        assert len(target_nodes) == 1, f"len={len(target_nodes)}"
        target_node = target_nodes[0]
        assert target_node

        assert self.make_connection(source_node, target_node)

        return NodeTransformType.DESTROY


class ReplaceMatMulWithGemm(Optimizer):
    def match(self, node: Node) -> bool:
        return node.op_type.startswith("MatMul")

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"FuseSqueeze on node {node.op_type} {node.name}")
        # we are targeting Squeeze
        node.op_type = "Gemm"

        return NodeTransformType.MODIFY
