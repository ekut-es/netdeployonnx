from collections import defaultdict
from typing import Any

import numpy as np
import onnx


def onnx_node_to_str(node: onnx.NodeProto) -> str:
    return f"{node.name}: {node.input} -> {node.output}" if node else "None"


def print_table(
    data_accessor=(lambda r, c: ""),
    COLUMN_WIDTH=10,  # noqa: N803
    COLUMNS=4,  # noqa: N803
    ROWS=16,  # noqa: N803
):
    """
    Print a table
    Args:
        data_accessor: a function that returns the data for a cell, row starts at 1,
            col starts at 0
        COLUMN_WIDTH: the width of each column
        COLUMNS: the number of columns
        ROWS: the number of rows
    """

    if COLUMN_WIDTH == -1:
        colw_per_row = [
            max(len(data_accessor(row, column)) for column in range(COLUMNS))
            for row in range(1, ROWS + 1)
        ]
        COLUMN_WIDTH = max(colw_per_row)  # noqa: N806

    def size_col(data):
        data = str(data)
        remaining_size = COLUMN_WIDTH - len(data)
        space_left = remaining_size // 2
        space_right = remaining_size - space_left
        return " " * space_left + data[:COLUMN_WIDTH] + " " * space_right

    def column_access(row, column, header=True):
        if header and (row == 0 or row == ROWS):
            return "-" * (COLUMN_WIDTH)
        else:
            return data_accessor(row, column)

    for row in range(ROWS + 1):
        cols = [size_col(column_access(row, column)) for column in range(COLUMNS)]
        line = "|".join([""] + cols + [""])
        print(line)


class Node:
    def __init__(self, node: onnx.NodeProto, graph: onnx.GraphProto):
        self.op_type: str = node.op_type
        self.name: str = node.name
        self.input: dict[str, Any] = {}
        self.output: dict[str, Any] = {}
        self.attributes: dict[str, Any | np.ndarray] = {}
        self._parse(node, graph)

    def _parse(self, node: onnx.NodeProto, graph: onnx):
        self.input = {
            input: self._get_io_value(NodeGrid.resolve_in_graph(graph, input))
            for input in node.input
        }
        self.output = {
            output: self._get_io_value(NodeGrid.resolve_in_graph(graph, output))
            for output in node.output
        }
        for attr in node.attribute:
            self.attributes[attr.name] = onnx.helper.get_attribute_value(attr)
            self.attributes[attr.name] = self._get_io_value(self.attributes[attr.name])

    def _get_io_value(self, value: Any) -> None | np.ndarray:
        if type(value) in [dict, list, int, float, type(None), str, bool, bytes]:
            return value
        if isinstance(value, onnx.ValueInfoProto):
            return None  # ?
        elif isinstance(value, onnx.TensorProto):
            return onnx.numpy_helper.to_array(value)
        elif isinstance(value, onnx.TypeProto):
            raise NotImplementedError("TypeProto")
        else:
            raise NotImplementedError(f"Unknown type {type(value)}")

    def onnx(self) -> onnx.NodeProto:
        onnx_attributes = [
            onnx.helper.make_attribute(
                key=attr_name,
                value=(
                    onnx.numpy_helper.from_array(attr_value)
                    if isinstance(attr_value, np.ndarray)
                    else attr_value
                ),
            )
            for attr_name, attr_value in self.attributes.items()
        ]
        node = onnx.NodeProto()
        node.op_type = self.op_type
        node.input.extend([name for name, value in self.input.items()])
        node.output.extend([name for name, value in self.output.items()])
        if self.name:
            node.name = self.name
        if onnx_attributes:
            node.attribute.extend(onnx_attributes)
        return node


class NodeGrid:
    def __init__(self, graph: onnx.GraphProto):
        self.graph = graph
        self.data = defaultdict(lambda: None)
        self.coords: dict[str, tuple[int, int]] = {}

    def __getitem__(self, key: tuple[int, int]):
        """
        Get a node from the grid
        Params:
            key: tuple[int, int] or str: x,y, starts at (0,0)
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.data[key[0], key[1]]
        else:
            raise KeyError(f"key {key} is not a tuple[int, int]")

    def __setitem__(self, key: tuple[int, int], value: Node):
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(value, Node):
                x, y = key
                self.data[x, y] = value
                for input in value.input:
                    if input not in self.coords:
                        self.coords[input] = (x, y - 0.5)
                for output in value.output:
                    if output not in self.coords:
                        self.coords[output] = (x, y + 0.5)
                self.coords[value.name] = key
            else:
                raise ValueError("value is not a Node")
        else:
            raise KeyError(f"key {key} is not a tuple[int, int]")

    def __contains__(self, key: tuple[int, int] | str):
        if isinstance(key, tuple) and len(key) == 2:
            return self.data[key[0], key[1]] is not None
        elif isinstance(key, str):
            return key in self.coords
        else:
            raise KeyError(f"key {key} is not a tuple[int, int]")

    def __delitem__(self, key: tuple[int, int]):
        raise NotImplementedError("__delitem__")

    @property
    def width(self):
        # TODO: optimize in the future? -> O(n^2)
        return max(x for x, y in self.data) + 1

    @property
    def height(self):
        # TODO: optimize in the future? -> O(n^2)
        return max(y for x, y in self.data) + 1

    def resolve(self, key: str) -> Any:
        return self.resolve_in_graph(self.graph, key)

    @classmethod
    def resolve_in_graph(cls, graph: onnx.GraphProto, key: str) -> Any:
        # now search for the key in the grap
        for inits in graph.initializer:
            if inits.name == key:
                return inits
        for inputs in graph.input:
            if inputs.name == key:
                return inputs
        for outputs in graph.output:
            if outputs.name == key:
                return outputs
        for nodes in graph.node:
            if nodes.name == key:
                return nodes
        # raise KeyError(f"key {key} not found in graph")
        # we dont want to raise because maybe its an intermediate output or input
        return None
