from pathlib import Path

import onnx

from netdeployonnx.devices.max78000.synthesizer.grid import Node


def test_node_to_onnx_and_back():
    # Test the import / export
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    graph = model.graph
    for node in graph.node:
        node_ir = Node(node, graph)
        for attrname, attrval in node_ir.attributes.items():
            assert attrval is not None

        assert node == node_ir.onnx()
