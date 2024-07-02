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
        assert len(node.output) == 1, "Conv should have only one output"

        node.op_type += "Relu"
        node.name = "/".join(node.name.split("/")[:-1] + [node.op_type])
        # delete the following relu node
        outputs = set()
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
        # now we need to replace the output of the node
        node.output = list(outputs)
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
        print(f"running on {node}")
        inputs = set()
        for source_node in self.source(node)():
            print("found node")
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
        node.op_type += "MaxPool"
        node.name = "/".join(node.name.split("/")[:-1] + [node.op_type])
        return NodeTransformType.MODIFY


class FuseConvReshape(Optimizer):
    """
    Fuse conv+reshape nodes from the graph
    """

    def match(self, node: Node) -> bool:
        # only match conv + relu if all the outputs are relu
        return node.op_type.startswith("Conv") and all(
            self.target(node).op_type.startswith("Reshape")
        )

    def run_transformation(self, node: Node) -> NodeTransformType:
        """
        Run on a conv layer, that is followed by a relu layer
        """
        assert len(node.input) == 3, "Conv should have 3 inputs: X, W, B"
        assert len(node.output) == 1, "Conv should have only one output"

        node.op_type += "Reshape"
        node.name = "/".join(node.name.split("/")[:-1] + [node.op_type])
        # delete the following relu node
        outputs = set()
        for target_node in self.target(node)():
            # we should only have one as node.output == 1
            if not target_node.op_type.startswith("Reshape"):
                # we should not be able to fuse that then?
                raise ValueError("We can only fuse if all outputs are Reshape")
            assert len(target_node.output) == 1, "Reshape should have only one output"
            outputs |= set(target_node.output)
            for attrname, attrval in target_node.attributes.items():
                newattr = f"_reshape_{attrname}"
                if newattr in node.attributes:
                    raise ValueError(f"Attribute {newattr} already exists in node")
                node.attributes[newattr] = attrval
            node.attributes["shape"] = target_node.input[1]  # input1 is shape
            target_node.deleted = True
        # now we need to replace the output of the node
        node.output = list(outputs)
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
        logger.info(f"running transformation on node {node.op_type} {node.name}")
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
        node.name = "/".join(node.name.split("/")[:-1] + ["Pass"])
        node.input = [good_input]
        node.output = [good_output]

        # we need to destroy the second clip node
        second_clip_node.deleted = True
        return NodeTransformType.MODIFY


class FuseSqueeze(Optimizer):
    """
    Eliminate quantization nodes from the graph
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
        logger.info(f"FuseSqueeze on node {node.op_type} {node.name}")
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
        node.name = "/".join(node.name.split("/")[:-1]) + "/Squeeze"
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
        logger.info(f"FuseSqueeze on node {node.op_type} {node.name}")
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
