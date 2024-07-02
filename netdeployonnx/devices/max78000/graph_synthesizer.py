import logging

import onnx

from netdeployonnx.devices.max78000.core import CNNx16Core
from netdeployonnx.devices.max78000.synthesizer.backend import (
    configure_layer_by_node,
)
from netdeployonnx.devices.max78000.synthesizer.grid import Node, NodeGrid
from netdeployonnx.devices.max78000.synthesizer.scheduler import ASAPScheduler

logger = logging.getLogger("root")


def grid_optimizer_maximize_utilization(grid: NodeGrid) -> NodeGrid:
    """
    Maximize the utilization of the grid
    """
    # if there are empty columns, we duplicate either x1, x2 or x4
    assert grid[1, 1] is None  # first row, second column is None
    for col in [1, 2, 3]:
        if 1:  # TODO: actually care about this
            # we can duplicate x1
            for row in range(grid.height):
                grid[col, row] = grid[0, row]
    assert grid[0, 1] is not None
    return grid


def grid_to_core(grid: NodeGrid) -> CNNx16Core:
    """
    Convert a grid to a core
    """
    core = CNNx16Core()
    assert grid.width <= 4, "grid is too wide"
    assert grid.height <= 16, "grid is too high"

    for quad in range(grid.width):
        shape = [1, 3, 32, 32]
        for layer in range(grid.height):
            node: onnx.NodeProto = grid[quad, layer]  # its x,y
            if node:
                shape = configure_layer_by_node(core[quad, layer], grid, node, shape)

    return core


def synthesize_to_grid(graph: onnx.GraphProto) -> NodeGrid:
    grid = synthesizer_draft(graph=graph)

    # run grid_optimizers
    for grid_optimizer in [grid_optimizer_maximize_utilization]:
        grid = grid_optimizer(grid)

    return grid


def synth_to_core_ir(graph: onnx.GraphProto) -> CNNx16Core:
    grid = synthesize_to_grid(graph)
    core = grid_to_core(grid)
    return core


def find_paths(  # noqa: C901
    graph: onnx.GraphProto, input: str, output: str
) -> list[Node]:
    """
    Find paths from input to output in a graph
    Args:
        graph: the graph to search in
        input: the input to start from
        output: the output to end at
    Returns:
        list of paths
    """
    # TODO:
    # rewrite this function because of complexity

    paths = []
    start_nodes = []
    end_nodes = []
    node_by_input = {}
    for node in graph.node:
        # print(node.input, node.output)
        if input in node.input:
            start_nodes.append(node)
        if output in node.output:
            end_nodes.append(node)
        for input_conn in node.input:
            node_by_input[input_conn] = node
    # add endnodes too
    for end_node in end_nodes:
        for output_conn in end_node.output:
            node_by_input[output_conn] = end_node

    for start_node in start_nodes:
        # dfs for one end node
        path: list[onnx.NodeProto] = []
        stack: list[onnx.NodeProto] = [start_node]
        while stack:
            node = stack.pop()  # pop without arguments = last element = stack
            path.append(node)
            for next_node in [node_by_input[output] for output in node.output]:
                if node in end_nodes:
                    # we found one of the end nodes
                    paths.append(path + [node])
                else:  # need to search deeper
                    stack.append(next_node)

    paths_without_duplicates: list[Node] = []
    for path in paths:
        # remove duplicates in one path
        seen = set()
        path = [
            Node(node, graph)
            for node in path
            if not (node.name in seen or seen.add(node.name))
        ]
        paths_without_duplicates.append(path)

    return paths_without_duplicates


def put_path_in_grid(
    grid: dict,
    path: list[Node],
    dependency_free_func: callable = None,
    start_x: int = 0,
    start_y: int = 0,
) -> int:
    """
    Put a path into a grid, starting at start_x, start_y
    Args:
        grid: the grid to put the path into

    """
    grid_width = 0

    scheduler_class = ASAPScheduler  # TODO: make this configurable
    last_x = start_x
    last_y = start_y

    unscheduled_nodes = list(path)
    while len(unscheduled_nodes):
        node, unscheduled_nodes = scheduler_class.select_predecessors_scheduled(
            grid,
            unscheduled_nodes,
            dependency_free_func=dependency_free_func,
        )

        # schedule
        x, y = scheduler_class.schedule(grid, node, previous=(last_x, last_y))
        if grid[x, y]:
            raise Exception("overwriting a layer?!")
        grid[x, y] = node
        grid_width = max(grid_width, x)
        last_x, last_y = x, y

    # print the schedule
    # print_table(
    #     data_accessor=lambda r, c: (
    #         f"{grid[c,r-1].op_type}[{grid[c,r-1].op_type}]" if grid[c, r - 1] else "~"
    #     ),
    #     COLUMN_WIDTH=-1,
    #     COLUMNS=last_x + 1,
    #     ROWS=last_y + 1,
    # )

    return grid_width


def fuse_paths(paths: list[list[Node]]) -> list[list[Node]]:
    """
    Fuses paths together, if possible
    Args:
        paths: list of paths to fuse
    Returns:
        list of fused paths
    """
    return paths  # TODO: remove, as we did not fuse anything
    # TODO: maybe a fused path is a tree?


def synthesizer_draft(graph: onnx.GraphProto) -> NodeGrid:
    """
    Synthesizes a graph into a grid
    Args:
        graph: the graph to synth to a grid / just before IR
    Returns:
        grid: the synthesized grid
    """
    # now how do we start

    # at first, we want to see the whole graph and its dimensions
    # so we want to place it into a grid, but the grid should be slim
    # it should be slim, so we can use column-replication

    # i think its best to have no optimizations at first and then later optimize
    grid = NodeGrid(graph)

    # we need to know the input and output of the graph
    inputs = [inp.name for inp in graph.input]
    outputs = [outp.name for outp in graph.output]

    # since we dont know anything about the graph, we cant assume anything,
    # and thats why we collect paths from all inputs to all outputs
    all_paths: list[Node] = []
    for input in inputs:
        for output in outputs:
            paths = find_paths(graph, input, output)
            if paths:
                all_paths.extend(paths)
    # now we have paths, and we can start to fuse them, if possible
    fused_paths = []

    # assume black box, and we somehow get our fused paths
    fused_paths = fuse_paths(all_paths)

    def check_connection(connection_name):
        """
        Check if a connection has a reason to be dependency_free
        hopefully any returns only
        """
        reason = None
        reasons = {
            # constants have no dependencies
            "is_constant": (lambda: connection_name.startswith("/Constant")),
            # inputs are always dependency free
            "is_input": (
                lambda: any(connection_name == input.name for input in graph.input)
            ),
            # initializers are also dependency free
            "is_initializer": (
                lambda: any(
                    connection_name == initializer.name
                    for initializer in graph.initializer
                )
            ),
        }
        for reason, check in reasons.items():
            if check():
                break
        return reason

    # then use the fused paths to place in a grid
    max_width = 0
    for path in fused_paths:
        path_grid_width = put_path_in_grid(
            grid,
            path,
            dependency_free_func=check_connection,
            start_x=max_width,
            start_y=0,
        )
        max_width = max(max_width, path_grid_width)

    # now every path should be layouted
    # we can now start to optimize the grid in another function
    return grid
