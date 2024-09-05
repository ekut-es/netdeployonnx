import abc
import enum
import logging

from .graph import Graph, Node

# checked graph optimizers:
# - github.com/onnx/optimizer (unfortunately not very useful, because c api and standalone tool)  # noqa: E501
# - networkx.org/documentation/stable/reference/algorithms/optimization.html
logger = logging.getLogger(__name__)


class ConditionCollector:
    supported_node_attrs = ["startswith", "endswith", "contains", "op_type"]

    def __init__(self, nodes: list[Node] = None, parent=None, condition=None):
        self.nodes = nodes
        self.parent = parent
        self.condition = condition

    def __getattr__(self, name):
        if name in self.supported_node_attrs:
            return ConditionCollector(parent=self, condition=name)
        raise AttributeError(f"ConditionCollector: {name}")

    def __call__(self, *args, **kwargs) -> list[Node]:
        # print("we got called!", args, kwargs, "with condition", self.condition)
        if self.condition is None:
            if self.nodes:
                return self.nodes
            elif self.parent:
                return self.parent(*args, **kwargs)
            else:
                # return empty list of nodes
                return []
        if self.condition in self.supported_node_attrs:
            # our nodes, we can operate directly?
            if self.nodes:
                nodes = self.nodes
            elif self.parent:
                # we cannot do it directly, but we can do it via the parent
                # print(self.parent)
                nodes = self.parent(*args, **kwargs)
                # print("nodes", nodes)
            else:
                raise AttributeError(f"ConditionCollector: {self.condition}")
            checkers = [
                getattr(node, self.condition)
                for node in nodes
                if hasattr(node, self.condition)
            ]

            def is_callable(x):
                return hasattr(x, "__call__")

            results = [
                checker(*args, **kwargs) if is_callable(checker) else checker
                for checker in checkers
            ]
            return results
        else:
            raise AttributeError(
                f"ConditionCollector: Unknown Condition{self.condition}"
            )

    def __repr__(self):
        has_parent = f"parent: {True}" if self.parent else ""
        has_condition = f"condition: {self.condition}" if self.condition else ""
        has_nodes = f"nodes: len={len(self.nodes)}" if self.nodes else ""
        return f"ConditionCollector: {has_parent} {has_condition} {has_nodes}"


class NodeTransformType(enum.Enum):
    # do nothing
    NO_CHANGE = 0  # has to be 0
    # destroy the node
    DESTROY = 1
    # modify the node
    MODIFY = 2


class Optimizer(abc.ABC):
    """
    Base class for all optimizers
        inspired by github.com/onnx/optimizer,
        f.ex. https://github.com/onnx/optimizer/blob/master/onnxoptimizer/passes/eliminate_nop_pad.h
    """

    def __init__(self): ...

    def get_pass_name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def match(self, node: Node) -> bool:
        raise NotImplementedError("match")

    @abc.abstractmethod
    def run_transformation(self, node: Node, graph: Graph) -> NodeTransformType:
        raise NotImplementedError("run_transformation")

    def run_on_graph(self, graph: Graph):
        changes = 0
        for node in graph:
            if not node.deleted and self.match(node):
                replacement_node: NodeTransformType = self.run_transformation(node)
                assert len(node.output) == 1, (
                    f"only one output supported, faulty optimizer "
                    f"{self.__class__.__name__}"
                )
                assert isinstance(
                    replacement_node, NodeTransformType
                ), f"transformer {self.get_pass_name()} did not return a valid NodeTransformType"  # noqa: E501
                if replacement_node == NodeTransformType.NO_CHANGE:
                    # skip
                    continue
                elif replacement_node == NodeTransformType.DESTROY:
                    node.deleted = True
                    changes += 1
                elif replacement_node == NodeTransformType.MODIFY:
                    changes += 1
                else:
                    raise NotImplementedError("run_on_graph")
        return changes

    def target(self, node: Node) -> ConditionCollector:
        return ConditionCollector(
            [
                node.graph.input_map[output]
                for output in node.output
                if output in node.graph.input_map
            ]
        )

    def source(self, node: Node) -> ConditionCollector:
        return ConditionCollector(
            [
                node.graph.output_map[input]
                for input in node.input
                if input in node.graph.output_map
            ]
        )

    def find_path(self, source: Node, target: Node) -> list[str]:
        # find the path from source to target
        import networkx as nx

        g = source.graph.networkx()
        try:
            path = nx.shortest_path(g, source, target)
            src = path[0]
            path_named = []
            for path_elem in path[1:]:
                # pick any connection that is in both src and path_elem
                path_named.append(
                    [output for output in src.output if output in path_elem.input][0]
                )
                src = path_elem
            return path_named
        except nx.NetworkXNoPath:
            return []

    def make_connection(self, source: Node, target: Node):
        # identify source output
        path = self.find_path(source, target)
        if path:
            relevant_outputs = [output for output in source.output if output in path]
            assert len(relevant_outputs) >= 1, "has to be atleast len 1"
            relevant_output_name = relevant_outputs[0]

            # identify target input
            relevant_inputs = [input for input in target.input if input in path]
            assert len(relevant_inputs) >= 1, "has to be atleast len 1"
            relevant_input_name = relevant_inputs[0]

            inp_list = list(target.input)
            assert relevant_input_name in inp_list
            inp_list[inp_list.index(relevant_input_name)] = relevant_output_name
            target.input = inp_list
        else:
            raise NotImplementedError(
                "what do we do when there is no path?" " create a new name?"
            )
        return True
