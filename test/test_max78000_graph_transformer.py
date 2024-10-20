import logging
from pathlib import Path

import onnx

from netdeployonnx.devices.max78000.graph_transformer import Graph, transform_graph
from netdeployonnx.devices.max78000.optimizer import (
    EliminateDanglingNodes,
    EliminatePassthrough,
    EliminateSqueeze,
    FuseBatchNorm,
    FuseClipQuantization,
    FuseConvMaxPool,
    FuseConvSqueeze,
    FuseFlatten,
    FuseGemmConvRelu,
    FuseReshape,
    FuseSqueeze,
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


def test_optimizer_eliminate_dangling_nodes():
    # Test the optimizer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    # Transform the graph
    transformed_graph = Graph(model.graph)
    node_count_before = len(transformed_graph)
    node_name_list = list(node.name for node in transformed_graph)
    changes = EliminateDanglingNodes().run_on_graph(transformed_graph)
    assert changes == 0

    assert node_count_before == len(transformed_graph)
    assert all(node.name in node_name_list for node in transformed_graph)


def test_optimizer_fused_squeeze():
    import numpy as np

    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node("Mul", ["pow", "div"], ["out"], name="Mul1"),
                onnx.helper.make_node(
                    "Div", ["input1", "constant1"], ["div"], name="Div1"
                ),
                onnx.helper.make_node(
                    "Pow", ["constant2", "constant3"], ["pow"], name="Pow1"
                ),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                ),
            ],
            [],
            initializer=[
                onnx.helper.make_tensor(
                    "constant1",
                    onnx.TensorProto.FLOAT,
                    dims=[],  # scalar
                    vals=np.array([3], dtype=np.float32).tobytes(),
                    raw=True,
                ),
                onnx.helper.make_tensor(
                    "constant2",
                    onnx.TensorProto.FLOAT,
                    dims=[],  # scalar
                    vals=np.array([2], dtype=np.float32).tobytes(),
                    raw=True,
                ),
                onnx.helper.make_tensor(
                    "constant3",
                    onnx.TensorProto.FLOAT,
                    dims=[],  # scalar
                    vals=np.array([4], dtype=np.float32).tobytes(),
                    raw=True,
                ),
            ],
        )
    )
    changes = FuseSqueeze().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == 3
    assert all(node.op_type != "Div" for node in transformed_graph)
    assert all(node.op_type != "Pow" for node in transformed_graph)
    assert any(node.op_type == "Squeeze" for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")
    actual_node = next(node for node in transformed_graph if node.op_type == "Squeeze")

    assert init_node.output[0] == actual_node.input[0]
    assert actual_node.output[0] == outro_node.input[0]  #

    assert np.isclose(actual_node.attributes["factor"], 2**4 / 3.0)


def test_optimizer_fuse_clip_quantization():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Clip", ["input1", "min", "max"], ["clip1"], name="Clip1"
                ),  # noqa
                onnx.helper.make_node(
                    "Clip", ["clip1", "min", "max"], ["out"], name="Clip2"
                ),  # noqa
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1", onnx.helper.make_tensor_type_proto(1, shape=None)
                )
            ],
            [],
        )
    )
    changes = FuseClipQuantization().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1
    assert all(node.op_type != "Clip1" for node in transformed_graph)
    assert any(node.op_type == "Pass" for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")
    actual_node = next(node for node in transformed_graph if node.op_type == "Pass")

    assert init_node.output[0] == actual_node.input[0]
    assert actual_node.output[0] == outro_node.input[0]


def test_optimizer_eliminate_pass():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node("Pass", ["input1"], ["out"], name="pass1"),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = EliminatePassthrough().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1
    assert all(node.op_type != "Pass" for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == outro_node.input[0]


def test_optimizer_eliminate_squeeze():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node("Squeeze", ["input1"], ["out"], name="Squeeze"),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = EliminateSqueeze().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1
    assert all(node.op_type != "Squeeze" for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == outro_node.input[0]


def test_fuse_conv_relu():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Conv", ["input1", "weights", "biases"], ["conv1"], name="Conv1"
                ),
                onnx.helper.make_node("Relu", ["conv1"], ["out"], name="Relu1"),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = FuseGemmConvRelu().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # relu should not be in graph
    assert all(node.op_type != "Relu1" for node in transformed_graph)
    # relu should not be in graph
    # we dont rename the type anymore.
    assert any(node.op_type == "Conv" for node in transformed_graph)
    assert any(node.name.endswith("Relu") for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    convrelu_node = next(node for node in transformed_graph if node.op_type == "Conv")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == convrelu_node.input[0]
    assert convrelu_node.output[0] == outro_node.input[0]


def test_fuse_conv_maxpool():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "MaxPool", ["input1"], ["maxpool"], name="maxpool1"
                ),
                onnx.helper.make_node(
                    "Conv", ["maxpool", "weights", "biases"], ["out"], name="Conv1"
                ),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = FuseConvMaxPool().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # relu should not be in graph
    assert all(node.op_type != "MaxPool" for node in transformed_graph)
    # relu should not be in graph
    # we dont rename the type anymore.
    assert any(node.op_type == "Conv" for node in transformed_graph)
    assert any(node.name.endswith("MaxPool") for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    convrelu_node = next(node for node in transformed_graph if node.op_type == "Conv")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == convrelu_node.input[0]
    assert convrelu_node.output[0] == outro_node.input[0]


def test_fuse_conv_squeeze():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Conv", ["input1", "weights", "biases"], ["conv"], name="conv1"
                ),
                onnx.helper.make_node("Squeeze", ["conv"], ["out"], name="squeeze1"),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = FuseConvSqueeze().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # relu should not be in graph
    assert all(node.op_type != "Squeeze" for node in transformed_graph)
    # relu should not be in graph
    assert any(node.op_type == "Conv" for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    conv_node = next(node for node in transformed_graph if node.op_type == "Conv")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == conv_node.input[0]
    assert conv_node.output[0] == outro_node.input[0]


def test_fuse_reshape():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Reshape", ["input1", "shape"], ["gemm"], name="reshape1"
                ),
                onnx.helper.make_node(
                    "Gemm", ["gemm", "weights", "biases"], ["out"], name="gemm1"
                ),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = FuseReshape().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # Reshape should not be in graph
    assert all(node.op_type != "Reshape" for node in transformed_graph)
    # Gemm should not be in graph
    assert any(node.op_type != "Gemm" for node in transformed_graph)
    # GemmReshape should be in graph
    # we dont rename anymore.
    assert any(node.op_type == "Gemm" for node in transformed_graph)
    assert any(node.name.endswith("Reshape") for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    fused_node = next(node for node in transformed_graph if node.op_type == "Gemm")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == fused_node.input[0]
    assert fused_node.output[0] == outro_node.input[0]


def test_fuse_flatten():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Flatten", ["input1", "shape"], ["gemm"], name="flatten1"
                ),
                onnx.helper.make_node(
                    "Gemm", ["gemm", "weights", "biases"], ["out"], name="gemm1"
                ),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [],
        )
    )
    changes = FuseFlatten().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # Reshape should not be in graph
    assert all(node.op_type != "Flatten" for node in transformed_graph)
    # Gemm should not be in graph
    assert any(node.op_type != "Gemm" for node in transformed_graph)
    # GemmReshape should be in graph
    # we dont rename anymore.
    assert any(node.op_type == "Gemm" for node in transformed_graph)
    assert any(node.name.endswith("Flatten") for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    fuse_node = next(node for node in transformed_graph if node.op_type == "Gemm")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == fuse_node.input[0]
    assert fuse_node.output[0] == outro_node.input[0]


def test_fuse_batchnorm():
    # Transform the graph
    transformed_graph = Graph(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Init", [], ["input1"], name="Initializer1"),
                onnx.helper.make_node(
                    "Conv", ["input1", "weights", "biases"], ["conv"], name="conv1"
                ),
                onnx.helper.make_node(
                    "BatchNormalization",
                    ["conv", "scale", "bias", "mean", "var"],
                    ["out", "mean", "var"],
                    name="BatchNormalization1",
                ),
                onnx.helper.make_node("Out", ["out"], [], name="Outro1"),
            ],
            "test",
            [
                onnx.helper.make_value_info(
                    "input1",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                )
            ],
            [
                onnx.helper.make_value_info(
                    "scale",
                    onnx.helper.make_tensor_type_proto(1, shape=None),
                ),
            ],
        )
    )
    changes = FuseBatchNorm().run_on_graph(transformed_graph)
    assert changes == 1
    assert len(transformed_graph) == len(transformed_graph.nodes) - 1  # relu is removed
    # Reshape should not be in graph
    assert all(node.op_type != "BatchNormalization" for node in transformed_graph)
    # Gemm should not be in graph
    assert any(node.op_type != "Gemm" for node in transformed_graph)
    # GemmReshape should be in graph
    # we dont rename anymore.
    assert any(node.op_type == "Conv" for node in transformed_graph)
    assert any(node.name.endswith("BatchNormalization") for node in transformed_graph)

    # is the input connected in both ways?
    init_node = next(node for node in transformed_graph if node.op_type == "Init")
    fuse_node = next(node for node in transformed_graph if node.op_type == "Conv")
    outro_node = next(node for node in transformed_graph if node.op_type == "Out")

    assert init_node.output[0] == fuse_node.input[0]
    assert fuse_node.output[0] == outro_node.input[0]


def test_fuse_mul_pow_factor():
    raise NotImplementedError()


def test_transform_graph():
    # Test the transformer
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    # Transform the graph
    transformed_graph = transform_graph(model.graph)

    # check that we removed the quantization nodes
    # FuseSqueeze
    assert all(not node.op_type.startswith("Mul") for node in transformed_graph.node)
    # EliminatePassthrough
    assert all(not node.op_type.startswith("Pass") for node in transformed_graph.node)

    # make new model
    model = onnx.helper.make_model(transformed_graph)
    onnx.save_model(model, data_folder / "cifar10_transformed.onnx")
