from collections.abc import Iterator

import networkx as nx
import numpy as np
import onnx


class Node:
    def __init__(self, graph: "Graph", node: onnx.NodeProto):
        self.graph = graph
        self.node = node
        self.op_type = node.op_type
        self.name = node.name
        self._input = [input for input in node.input]
        self._output = [output for output in node.output]
        self.attributes = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        self.doc_string = node.doc_string
        self.domain = node.domain
        self.deleted = False

    @property
    def source_nodes(self) -> Iterator["Node"]:
        self.graph.update_io_maps()
        for input in self.input:
            if input and input in self.graph.output_map:
                yield self.graph.output_map[input]

    @property
    def target_nodes(self) -> Iterator["Node"]:
        self.graph.update_io_maps()
        for output in self.output:
            if output and output in self.graph.input_map:
                yield self.graph.input_map[output]

    @property
    def input_map(self) -> dict:
        # return a subset
        # TODO: check if that is computational expensive
        return {
            input: self.graph.input_map[input]
            for input in self.input
            if input and input in self.graph.input_map
        }

    @property
    def output_map(self) -> dict:
        # return a subset
        # TODO: check if that is computational expensive
        return {
            output: self.graph.output_map[output]
            for output in self.output
            if output and output in self.graph.output_map
        }

    @property
    def input(self) -> list:
        return self._input

    @input.setter
    def input(self, value: list):
        # update the mappings
        removed = set(self._input) - set(value)
        added = set(value) - set(self._input)
        for remove in removed:
            if remove:
                del self.graph.input_map[remove]
        for add in added:
            if add:
                self.graph.input_map[add] = self
        self._input = value

    @property
    def output(self) -> list:
        return self._output

    @output.setter
    def output(self, value: list):
        # update the mappings
        removed = set(self._output) - set(value)
        added = set(value) - set(self._output)
        for remove in removed:
            if remove:
                del self.graph.output_map[remove]
        for add in added:
            if add:
                self.graph.output_map[add] = self
        self._output = value

    def onnx(self) -> onnx.NodeProto:
        node = onnx.helper.make_node(
            op_type=self.op_type,
            name=self.name,
            inputs=self.input,
            outputs=self.output,
            doc_string=self.doc_string,
            domain=self.domain,
            **self.attributes,
        )
        return node

    def __repr__(self) -> str:
        return f"Node {self.name}:{self.op_type}"


class Graph:
    def __init__(self, graph: onnx.GraphProto):
        # init graph
        self.input_map = {}
        self.output_map = {}

        self.nodes = set(Node(self, node) for node in graph.node)
        self.name = graph.name
        self.input = {
            input.name: input.type for input in graph.input
        }  # TODO: check the ValueInfoProto
        self.output = {output.name: output.type for output in graph.output}
        self.initializers: dict[str, tuple[int:tensor_dtype, tuple, bytes]] = {
            initializer.name: (
                initializer.data_type,
                initializer.dims,
                initializer.raw_data,
            )
            # its a is a collection of TensorProto
            for initializer in graph.initializer
        }
        # self.doc_string = graph.doc_string
        self.value_info = set(graph.value_info)
        self.sparse_initializer = set(graph.sparse_initializer)

        # TODO: fix this
        self.update_io_maps()
        self.maingraph = set()
        self.update_maingraph()

    def update_io_maps(self):
        self.input_map: dict[str, Node] = {}
        self.output_map: dict[str, Node] = {}
        for nodex in self:
            for node_input in nodex.input:
                self.input_map[node_input] = nodex
            for node_output in nodex.output:
                self.output_map[node_output] = nodex

        for input in self.input:
            if input not in self.output_map:  # graph inputs are outputs for nodes
                self.output_map[input] = Node(
                    self,
                    onnx.helper.make_node(
                        "INPUT", inputs=[], outputs=[input], name=input
                    ),
                )
        for output in self.output:
            if output not in self.input_map:  # graph outputs are inputs for nodes
                self.input_map[output] = Node(
                    self,
                    onnx.helper.make_node(
                        "OUTPUT", inputs=[output], outputs=[], name=output
                    ),
                )

    def __iter__(self) -> Iterator[Node]:
        for node in self.nodes:
            if node.deleted:
                continue
            yield node

    def __len__(self) -> int:
        return len([node for node in self.nodes if not node.deleted])

    def deleted_nodes(self) -> Iterator[Node]:
        for node in self.nodes:
            if node.deleted:
                yield node

    def update_maingraph(self):
        # we need to find the main graph
        self.maingraph = set()
        # instead of finding any input, we just pick a random node
        # and assume no conv or gemm would be removed
        start_node = next(
            # TODO: document this
            node
            for node in self.nodes
            if node.op_type.startswith("Conv") or node.op_type.startswith("Gemm")
        )

        # assert that there are no duplicate names
        assert len(self.nodes) == len(
            set([node.name for node in self.nodes])
        ), "Duplicate names in graph"

        # now bfs to find all nodes
        queue = [start_node]
        while queue:
            current_node = queue.pop(0)
            self.maingraph |= set([current_node.name])
            for next_node in current_node.target_nodes:
                # no cycles
                if next_node.name not in self.maingraph:
                    # logging.debug(f"Adding {next_node.name} to maingraph")
                    queue.append(next_node)
            # we also need to check inputs
            for next_node in current_node.source_nodes:
                if next_node.name not in self.maingraph:
                    # logging.debug(f"Adding {next_node.name} to maingraph")
                    queue.append(next_node)

    def onnx(self) -> onnx.GraphProto:
        nodes = [node.onnx() for node in self.nodes if not node.deleted]

        digraph = nx.DiGraph()
        for node in nodes:
            digraph.add_node(node.name, node=node)
            for input in node.input:
                digraph.add_edge(input, node.name)
            for output in node.output:
                digraph.add_edge(node.name, output)
        nodes_toposorted_ = nx.topological_sort(digraph)
        nodes_toposorted = []

        # runtime is shit, but dont care
        for node_name in nodes_toposorted_:
            for node in nodes:
                if node.name == node_name:
                    nodes_toposorted.append(node)

        graph = onnx.helper.make_graph(
            nodes=nodes_toposorted,
            name="InspectableONNXGraph - INVALID",
            inputs=[
                onnx.helper.make_value_info(input, type)
                for input, type in self.input.items()
            ],
            outputs=[
                onnx.helper.make_value_info(output, type)
                for output, type in self.output.items()
            ],
            initializer=[
                onnx.helper.make_tensor(name, type, dims, raw_data, raw=True)
                for name, (type, dims, raw_data) in self.initializers.items()
            ],
            doc_string="just inspection",
            value_info=[],
            sparse_initializer=[],
        )
        return graph

    @staticmethod
    def resolve_input_name(  # noqa: C901
        onnxgraph: "onnx.GraphProto", input_name: str, visited_nodes: list[str] = []
    ) -> np.ndarray:
        def fix_shape(weights_shape, arr_size_flattened=0):
            if len(weights_shape) == 1:
                # assume 1D weights
                ret = [1, weights_shape[0], 1, 1]
            elif len(weights_shape) == 2:
                # assume 2D weights, now we should have [out_channels, in_channels]
                ret = list(weights_shape) + [1, 1]
            elif len(weights_shape) == 3:
                # this is either [out, in, shape] + [shape]
                # or [1, out, in, shape]
                if np.prod(weights_shape) == arr_size_flattened:
                    ret = [1] + list(weights_shape)
                else:
                    ret = list(weights_shape) + [weights_shape[-1]]
            elif len(weights_shape) == 4:
                ret = weights_shape
            else:
                ret = [1, 1, 1, 1]
            # logging.debug(f"fixing shape from {weights_shape} to {ret}")
            return weights_shape

        if input_name is None:
            return None
        # Resolve input name
        for initializer in onnxgraph.initializer:
            if initializer.name == input_name:
                arr = onnx.numpy_helper.to_array(initializer)
                weights_shape = arr.shape
                # why do we even need to fix the shape? i think its because of spox (hannah)
                # see 12e0537a588a327d0d9143090a09c56716732b6a
                new_shape = fix_shape(weights_shape, np.prod(weights_shape))
                # should only add 1,1 and therefore not change the shape
                return np.reshape(arr, new_shape)
        for node in onnxgraph.input:
            if node.name == input_name:
                return np.zeros(
                    fix_shape([d.dim_value for d in node.type.tensor_type.shape.dim])
                )
        for node in onnxgraph.node:
            if input_name in node.output:
                if node.op_type == "Constant":
                    attributes = {attr.name: attr for attr in node.attribute}
                    if "value" in attributes:
                        t = attributes["value"].t
                        return np.zeros(fix_shape(t.dims))
        return np.zeros([1, 1, 1, 1])

    def nxgraph_for_ai8x(self) -> nx.DiGraph:  # noqa: C901
        # TODO: merge later with networkx
        # Initialize a directed graph
        nx_graph = nx.DiGraph()
        onnxgraph = self.onnx()
        # Add nodes and edges
        for node in onnxgraph.node:
            kwargs = {}
            if node.op_type.startswith("Conv") or node.op_type.startswith("Gemm"):
                node_input = list(node.input)
                kwargs["weights"] = self.resolve_input_name(
                    onnxgraph, node_input[1] if len(node_input) >= 2 else None
                )
                assert kwargs["weights"] is not None, "weights is none"
                assert len(kwargs["weights"].shape) in [2, 3, 4], (
                    f"Invalid weights shape {kwargs['weights'].shape} "
                    f"for node {node.name}"
                )
                kwargs["bias"] = self.resolve_input_name(
                    onnxgraph, node_input[2] if len(node_input) >= 3 else None
                )
                kwargs["input"] = self.resolve_input_name(
                    onnxgraph, node_input[0] if len(node_input) >= 1 else None
                )
                for i, (attribute_name, attribute) in enumerate(
                    (attr.name, onnx.helper.get_attribute_value(attr))
                    for attr in node.attribute
                ):
                    if attribute_name in kwargs:
                        raise ValueError(f"Duplicate attribute name {attribute_name}")
                    kwargs[attribute_name] = attribute

            nx_graph.add_node(node.name, op_type=node.op_type, name=node.name, **kwargs)
            for input_name in node.input:
                nx_graph.add_edge(input_name, node.name)
            for output_name in node.output:
                nx_graph.add_edge(node.name, output_name)
        return nx_graph

    def networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for node in self:
            g.add_node(node)
            for target in node.target_nodes:
                g.add_edge(node, target, type="target")
            for source in node.source_nodes:
                g.add_edge(source, node, type="source")
        return g

    def get_const_value(self, name):
        import numpy as np

        if name in self.initializers:
            # return the value, which should be an extracted tensor
            tensortype, dims, data = self.initializers[name]
            if not dims or len(dims) == 1:  # means scalar
                extracted = np.frombuffer(data, dtype=np.float32)
                if len(extracted) >= 1:
                    return extracted[0]
                raise NotImplementedError(f"get_const_value {tensortype}")
            else:
                raise NotImplementedError(
                    f"get_const_value('{name}') has found a "
                    "bigger dimension than expected (bigger than 1)"
                )
        elif name in self.output_map:
            # maybe some node outputs it
            node = self.output_map[name]
            if node.op_type == "Constant":
                value = node.attributes["value"]
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(value.data_type)
                if np_dtype == np.dtype("float32"):
                    extracted = np.frombuffer(value.raw_data, dtype=np.float32)
                    if len(extracted) >= 1:
                        return extracted[0]
                    raise NotImplementedError(f"get_const_value {np_dtype}")
                else:
                    raise NotImplementedError(f"get_const_value {np_dtype}")
            if node.op_type == "Identity":
                return self.get_const_value(node.input[0])
        else:
            raise ValueError(f"Constant {name} not found")
