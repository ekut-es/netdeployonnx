import logging
from pathlib import Path
from typing import Any

import onnx
import pytest

from netdeployonnx.devices.max78000.graph_transformer import (
    Graph,
    transform_graph,
)
from netdeployonnx.devices.max78000.optimizer import (
    Augment_Conv_Kernelshape,
    Augment_Conv_WeightsBias,
    EliminateBatchNorm,
    EliminatePassthrough,
    FuseBatchNorm,
    FuseClipQuantization,
    FuseConvMaxPool,
    FuseGemmConvRelu,
    FuseQuantizeDequantizeLinear,
    FuseReshape,
    FuseSqueeze,
    ReplaceMatMulWithGemm,
)

logging.basicConfig(
    level=logging.INFO,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


def io_equal(orig, new):
    assert orig.name == new.name, f"{orig.name} != {new.name}"
    assert orig.type == new.type, f"{orig.type} != {new.type}"
    assert orig.doc_string == new.doc_string, f"{orig.doc_string} != {new.doc_string}"
    # assert orig.domain == new.domain, f"{orig.domain} != {new.domain}"
    return True


def node_equal(orig, new):
    assert orig.name == new.name, f"{orig.name} != {new.name}"
    assert orig.op_type == new.op_type, f"{orig.op_type} != {new.op_type}"
    assert orig.doc_string == new.doc_string, f"{orig.doc_string} != {new.doc_string}"

    # assert the inputs are the same
    assert len(orig.input) == len(new.input), f"{orig.input} != {new.input}"
    for i, input in enumerate(orig.input):
        assert input == new.input[i], f"{input} != {new.input[i]}"
    # assert the outputs are the same
    assert len(orig.output) == len(new.output), f"{orig.output} != {new.output}"
    for i, output in enumerate(orig.output):
        assert output == new.output[i], f"{output} != {new.output[i]}"
    # assert the attributes are the same
    assert len(orig.attribute) == len(
        new.attribute
    ), f"{orig.attribute} != {new.attribute}"
    for i, attr in enumerate(orig.attribute):
        assert attr == new.attribute[i], "attribute wrong"

    return True


def initializer_equal(orig, new):
    assert orig.name == new.name, f"{orig.name} != {new.name}"
    assert orig.data_type == new.data_type, f"{orig.data_type} != {new.data_type}"
    assert orig.doc_string == new.doc_string, f"{orig.doc_string} != {new.doc_string}"
    assert orig.raw_data == new.raw_data, f"{orig.raw_data} != {new.raw_data}"
    return True


def test_from_onnx_and_back():
    # Test the import / export
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    graph = model.graph
    conv_graph: onnx.GraphProto = Graph(graph).onnx()

    # assert the inputs are the same
    for i, input in enumerate(graph.input):
        assert io_equal(input, conv_graph.input[i])

    # assert the outputs are the same
    for i, output in enumerate(graph.output):
        assert io_equal(output, conv_graph.output[i])

    # assert the nodes are the same
    for i, node in enumerate(graph.node):
        called = False
        for j, conv_node in enumerate(conv_graph.node):
            if node.name == conv_node.name:
                assert node_equal(node, conv_node)
                called = True
                break
        if not called:
            assert False, f"Node {node.name} not found in new graph"
    if i != len(conv_graph.node) - 1:
        assert False, "not equal length in new graph"

    # assert the initializers
    for i, init in enumerate(graph.initializer):
        assert initializer_equal(init, conv_graph.initializer[i])

    model = onnx.helper.make_model(conv_graph)
    onnx.save_model(model, data_folder / "cifar10_orig.onnx")


@pytest.mark.skip("for now, remove me")
def test_transform_graph():
    # Test the transformer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    # Transform the graph
    transformed_graph_ = transform_graph(model.graph)

    # check that we removed the quantization nodes
    # FuseSqueeze
    assert all(not node.op_type.startswith("Mul") for node in transformed_graph_.node)
    # EliminatePassthrough
    assert all(not node.op_type.startswith("Pass") for node in transformed_graph_.node)

    # make new model
    model = onnx.helper.make_model(transformed_graph_)
    onnx.save_model(model, data_folder / "cifar10_transformed.onnx")


def create_graph(node_list: list[dict[str, Any]]):
    nodes = []
    for node_dict in node_list:
        create_args = {}
        create_args.update(node_dict)
        node = onnx.helper.make_node(**create_args)
        nodes.append(node)

    return Graph(
        onnx.helper.make_graph(
            nodes=nodes,
            name="test_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [1, 3, 32, 32]
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, [1, 10]
                )
            ],
        )
    )


@pytest.mark.parametrize(
    "graph, should_match",
    [
        (
            create_graph(
                [
                    dict(
                        op_type="Conv",
                        inputs=["input", "W", "B"],
                        outputs=["output"],
                        name="conv1",
                    ),
                ]
            ),
            False,
        ),
        (
            create_graph(
                [
                    dict(
                        op_type="Conv",
                        inputs=["input"],
                        outputs=["output"],
                        name="conv1",
                    ),
                ]
            ),
            True,
        ),
    ],
)
def test_Augment_Conv_WeightsBias(graph: Graph, should_match: bool):  # noqa: N802
    changes = Augment_Conv_WeightsBias().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes > 0
        assert len(graph.nodes) == 1
        assert len(list(graph.nodes)[0].input) == 3
        assert list(graph.nodes)[0].input[1] == "conv1_weight"


@pytest.mark.parametrize("graph, should_match", [])
def test_Augment_Conv_Kernelshape(graph: Graph, should_match: bool):  # noqa: N802
    changes = Augment_Conv_Kernelshape().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()


@pytest.mark.parametrize(
    "graph, should_match",
    [
        (
            create_graph(
                [
                    dict(
                        op_type="Mul",
                        inputs=["pow", "div"],
                        outputs=["out"],
                        name="Mul1",
                    ),
                    dict(
                        op_type="Div",
                        inputs=["input", "constant1"],
                        outputs=["div"],
                        name="Div1",
                    ),
                    dict(
                        op_type="Pow",
                        inputs=["constant2", "constant3"],
                        outputs=["output"],
                        name="Pow1",
                    ),
                ]
            ),
            True,
        ),
    ],
)
def test_FuseSqueeze(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseSqueeze().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == 3
        assert all(node.op_type != "Div" for node in graph)
        assert all(node.op_type != "Pow" for node in graph)
        assert any(node.op_type == "Squeeze" for node in graph)
        assert len(graph.nodes) == 1
        assert len(list(graph.nodes)[0].input) == 3
        assert list(graph.nodes)[0].input[1] == "conv1_weight"


@pytest.mark.parametrize("graph, should_match", [])
def test_EliminatePassthrough(graph: Graph, should_match: bool):  # noqa: N802
    changes = EliminatePassthrough().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == len(graph.nodes) - 1
        assert all(node.op_type != "Pass" for node in graph)

        # is the input connected in both ways?
        init_node = next(node for node in graph if node.op_type == "Init")
        outro_node = next(node for node in graph if node.op_type == "Out")

        assert init_node.output[0] == outro_node.input[0]


@pytest.mark.parametrize("graph, should_match", [])
def test_EliminateBatchNorm(graph: Graph, should_match: bool):  # noqa: N802
    changes = EliminateBatchNorm().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseBatchNorm(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseBatchNorm().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseQuantizeDequantizeLinear(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseQuantizeDequantizeLinear().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseClipQuantization(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseClipQuantization().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == len(graph.nodes) - 1
        assert all(node.op_type != "Clip1" for node in graph)
        assert any(node.op_type == "Pass" for node in graph)

        # is the input connected in both ways?
        init_node = next(node for node in graph if node.op_type == "Init")
        outro_node = next(node for node in graph if node.op_type == "Out")
        actual_node = next(node for node in graph if node.op_type == "Pass")

        assert init_node.output[0] == actual_node.input[0]
        assert actual_node.output[0] == outro_node.input[0]


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseConvSqueeze(graph: Graph, should_match: bool):  # noqa: N802
    changes = Augment_Conv_Kernelshape().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseConvRelu(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseGemmConvRelu().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == len(graph.nodes) - 1  # relu is removed
        # relu should not be in graph
        assert all(node.op_type != "Relu1" for node in graph)
        # relu should not be in graph
        assert any(node.op_type == "ConvRelu" for node in graph)

        # is the input connected in both ways?
        init_node = next(node for node in graph if node.op_type == "Init")
        convrelu_node = next(node for node in graph if node.op_type == "ConvRelu")
        outro_node = next(node for node in graph if node.op_type == "Out")

        assert init_node.output[0] == convrelu_node.input[0]
        assert convrelu_node.output[0] == outro_node.input[0]


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseConvMaxPool(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseConvMaxPool().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == len(graph.nodes) - 1  # relu is removed
        # relu should not be in graph
        assert all(node.op_type != "MaxPool" for node in graph)
        # relu should not be in graph
        assert any(node.op_type == "ConvMaxPool" for node in graph)

        # is the input connected in both ways?
        init_node = next(node for node in graph if node.op_type == "Init")
        convrelu_node = next(node for node in graph if node.op_type == "ConvMaxPool")
        outro_node = next(node for node in graph if node.op_type == "Out")

        assert init_node.output[0] == convrelu_node.input[0]
        assert convrelu_node.output[0] == outro_node.input[0]


@pytest.mark.parametrize("graph, should_match", [])
def test_FuseReshape(graph: Graph, should_match: bool):  # noqa: N802
    changes = FuseReshape().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        assert changes == 1
        assert len(graph) == len(graph.nodes) - 1  # relu is removed
        # Reshape should not be in graph
        assert all(node.op_type != "Reshape" for node in graph)
        # Gemm should not be in graph
        assert any(node.op_type != "Gemm" for node in graph)
        # GemmReshape should be in graph
        assert any(node.op_type == "GemmReshape" for node in graph)

        # is the input connected in both ways?
        init_node = next(node for node in graph if node.op_type == "Init")
        fuse_node = next(node for node in graph if node.op_type == "GemmReshape")
        outro_node = next(node for node in graph if node.op_type == "Out")

        assert init_node.output[0] == fuse_node.input[0]
        assert fuse_node.output[0] == outro_node.input[0]


@pytest.mark.parametrize("graph, should_match", [])
def test_ReplaceMatMulWithGemm(graph: Graph, should_match: bool):  # noqa: N802
    changes = ReplaceMatMulWithGemm().run_on_graph(graph)

    if not should_match:
        assert changes == 0
    else:
        raise NotImplementedError()
