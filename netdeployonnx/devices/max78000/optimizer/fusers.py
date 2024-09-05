from .graph import Node
from .optimizer import NodeTransformType, Optimizer, logger


class FuseConvRelu(Optimizer):
    """
    Fuse conv+relu nodes from the graph
    """

    def match(self, node: Node) -> bool:
        # only match conv + relu if all the outputs are relu
        return node.op_type.startswith("Conv") and all(
            self.target(node).op_type.startswith("Relu")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a conv layer, that is followed by a relu layer
        """
        assert len(node.input) == 3, "Conv should have 3 inputs: X, W, B"
        assert len(node.output) == 1, (
            f"Conv should have only one output "
            f"and has {len(node.output)} [{node.name}]"
        )

        # node.op_type += "Relu"
        node.name = "/".join(node.name.split("/")[:] + ["_"] + ["Relu"])
        # delete the following relu node
        outputs = set()
        # print("successos", node.graph.networkx().successors(node.name))

        for target_node in self.target(node)():
            if target_node.deleted:
                continue
            if not target_node.op_type.startswith("Relu"):
                # we should not be able to fuse that then?
                raise ValueError("We can only fuse if all outputs are Relu")
            assert len(target_node.output) == 1, "Relu should have only one output"
            outputs |= set(target_node.output)
            for attrname, attrval in target_node.attributes.items():
                newattr = f"_relu_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            target_node.deleted = True
        if list(outputs) == 0:
            raise ValueError("We should have found a relu node")
        # now we need to replace the output of the node
        node.output = list(outputs)
        assert len(node.output) == 1, f"outputs={node.output}"
        node.attributes["activation"] = "relu"
        return NodeTransformType.MODIFY


class FuseGemmRelu(Optimizer):
    """
    Fuse Gemm+relu nodes from the graph
    """

    def match(self, node: Node) -> bool:
        # only match gemm + relu if all the outputs are relu
        return node.op_type.startswith("Gemm") and all(
            self.target(node).op_type.startswith("Relu")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a gemm layer, that is followed by a relu layer
        """
        assert len(node.input) == 3, "Gemm should have 3 inputs: X, W, B"
        assert len(node.output) == 1, (
            f"Gemm should have only one output "
            f"and has {len(node.output)} [{node.name}]"
        )
        node.name = "/".join(node.name.split("/")[:] + ["_"] + [node.op_type])
        # delete the following relu node
        outputs = set()
        assert len(self.target(node)()) == 1, (
            f"len={len(self.target(node)())}, {list(self.target(node)())}, "
            f"outputs = {node.output}"
        )
        for target_node in self.target(node)():
            if not target_node.op_type.startswith("Relu"):
                # we should not be able to fuse that then?
                raise ValueError("We can only fuse if all outputs are Relu")
            assert len(target_node.output) == 1, "Relu should have only one output"
            outputs |= set(target_node.output)
            for attrname, attrval in target_node.attributes.items():
                newattr = f"_relu_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            target_node.deleted = True
            break
        # now we need to replace the output of the node
        node.output = list(outputs)
        assert len(node.output) == 1, f"outputs={node.output}"
        node.attributes["activation"] = "relu"
        return NodeTransformType.MODIFY


class FuseConvMaxPool(Optimizer):
    """
    Fuse conv+maxpool nodes from the graph
    """

    def match(self, node: Node) -> bool:
        # only match conv + relu if all the outputs are relu
        cond = self.source(node).op_type.startswith("MaxPool")
        return node.op_type.startswith("Conv") and len(cond) > 0 and all(cond)

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a conv layer, that is followed by a relu layer
        """
        assert len(node.input) == 3, (
            "Conv should have 3 inputs: X, W, B" f" but has {node.input}"
        )
        assert len(node.output) == 1, "Conv should have only one output"

        # delete the relu node
        logger.debug(f"running on {node}")
        inputs = set()
        for source_node in self.source(node)():
            logger.debug("found node")
            # we should only have one as node.input == 1
            if not source_node.op_type.startswith("MaxPool"):
                # we should not be able to fuse that then?
                raise ValueError("We can only fuse if the input are MaxPool")

            assert len(source_node.input) == 1, "MaxPool should have only one input"
            inputs |= set(source_node.input)
            for attrname, attrval in source_node.attributes.items():
                newattr = f"_maxpool_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            source_node.deleted = True
        # now we need to replace the input of the node
        assert len(inputs) == 1, f"inputs={inputs}"
        node.input = list(inputs) + node.input[1:]
        # node.op_type += "MaxPool" # we dont change the type
        node.name = "/".join(node.name.split("/")[:] + ["_"] + ["MaxPool"])
        assert len(node.output) == 1, f"outputs={node.output}"
        return NodeTransformType.MODIFY


class FuseReshape(Optimizer):
    """
    Fuse reshape nodes from the graph
    """

    def match(self, node: Node) -> bool:
        # only match Gemm + relu if all the outputs are relu
        return node.op_type.startswith("Gemm") and (
            all(self.source(node).op_type.startswith("Reshape"))
            or all(self.source(node).op_type.startswith("Flatten"))
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a Gemm layer, that is followed by a relu layer
        """
        assert len(node.input) in [2, 3], "Conv/Gemm should have 2 or 3 inputs: X, W, B"
        assert len(node.output) == 1, "Conv/Gemm should have only one output"

        # delete the following relu node
        inputs = list(node.input)
        for source_node in self.source(node)():
            if source_node.deleted:
                continue

            # node.op_type += "Reshape"
            node.name = "/".join(node.name.split("/")[:] + ["_"] + ["Reshape"])
            # we should only have one as node.output == 1
            if not source_node.op_type.startswith(
                "Reshape"
            ) and not source_node.op_type.startswith("Flatten"):
                # we should not be able to fuse that then?
                raise ValueError("We can only fuse if all inputs are Reshape/Flatten")
            assert len(source_node.output) == 1, "Reshape should have only one output"
            inputs[0] = source_node.input[0]
            for attrname, attrval in source_node.attributes.items():
                newattr = {
                    "Reshape": f"_reshape_{attrname}",
                    "Flatten": f"_flatten_{attrname}",
                }[source_node.op_type]
                if newattr in source_node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            if source_node.op_type.startswith("Reshape"):
                node.attributes["_reshape__shape"] = source_node.input[
                    1
                ]  # input1 is shape
            elif source_node.op_type.startswith("Flatten"):
                node.attributes["_flatten__axis"] = source_node.attributes["axis"]

            for supersource in self.source(source_node)():
                idx = supersource.output.index(source_node.input[0])
                supersource.output[idx] = node.input[0]
                break
            source_node.deleted = True

            # now we need to replace the output of the node
            node.inputs = list(inputs)
            return NodeTransformType.MODIFY
            break

        assert len(node.output) == 1, f"outputs={node.output}"  # gemm has only 1 output
        return NodeTransformType.NO_CHANGE


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


class FuseClipQuantization(Optimizer):
    """
    Fuse clip quantization from the graph
    """

    def match(self, node: Node) -> bool:
        return node.op_type.startswith("Clip") and any(
            self.target(node).op_type.startswith("Clip")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        logger.debug(f"running transformation on node {node.op_type} {node.name}")
        # we are targeting the first clip
        # so we have one output guaranteed
        assert len(node.output) == 1
        # and we are guaranteed to have two or three inputs
        assert len(node.input) in [2, 3]
        # we need to find the second clip node
        target_nodes = self.target(node)()
        second_clip_node = next(
            node for node in target_nodes if node.op_type.startswith("Clip")
        )

        previous_nodes = self.source(node)()
        previous_outputs: set[str] = {
            output for node in previous_nodes for output in node.output
        }
        # we need to find which is the good input, which is one of the outputs of our
        # previous node
        good_input = next(input for input in node.input if input in previous_outputs)
        assert good_input, f"good_input={good_input}"
        good_output = list(second_clip_node.output)[0]
        assert good_output, f"good_output={good_output}"

        # now we change this node
        node.op_type = "Pass"
        node.name = "/".join(node.name.split("/")[:] + ["_"] + ["Pass"])
        node.input = [good_input]
        node.output = [good_output]

        # we need to destroy the second clip node
        second_clip_node.deleted = True
        return NodeTransformType.MODIFY


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
