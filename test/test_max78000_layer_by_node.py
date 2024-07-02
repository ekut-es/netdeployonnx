import unittest.mock as mock
from pathlib import Path

import numpy as np
import onnx
import pytest

from netdeployonnx.devices.max78000.core import CNNx16_Layer, CNNx16Core
from netdeployonnx.devices.max78000.synthesizer.backend import (
    configure_layer_by_node,
)
from netdeployonnx.devices.max78000.synthesizer.grid import Node

from .test_max78000_graph_synthesis import make_kernel_data_per_layer


@pytest.fixture(scope="function")
def core():
    return CNNx16Core()


@pytest.fixture(scope="function")
def layer(core) -> CNNx16_Layer:
    return core[0, 0]


@pytest.fixture(scope="function")
def grid():
    grd = mock.MagicMock()
    return grd


def make_testnode(
    nodetype: str, inputs: list[str], outputs: list[str], **kwargs
) -> Node:
    onnx_node = mock.MagicMock()
    onnx_graph = mock.MagicMock()

    def get_attr(name, val):
        mm = mock.MagicMock()
        mm.name = name
        mm.ref_attr_name = False
        mm.type = onnx.AttributeProto.TENSOR
        mm.t = val
        return mm

    def get_iomock_unpack(name: str, val):
        iomock = mock.MagicMock()
        iomock.name = name
        iomock.val = val
        return iomock

    onnx_node.op_type = nodetype
    onnx_node.input = [name for name, val in kwargs.items() if name in inputs]
    onnx_node.output = [name for name, val in kwargs.items() if name in outputs]
    onnx_node.attribute = [
        get_attr(name, get_iomock_unpack(name, val))
        for name, val in kwargs.items()
        if name not in outputs and name not in inputs
    ]

    def _get_io_value(self, iomock) -> None | np.ndarray:
        return iomock.val

    with mock.patch(
        "netdeployonnx.devices.max78000.synthesizer.grid.Node._get_io_value",
        _get_io_value,
    ) as _get_io_value:
        onnx_graph.input = [
            get_iomock_unpack(name, val) for name, val in kwargs.items()
        ]
        node = Node(onnx_node, onnx_graph)
    return node


def make_cifar10_refnet(_kernel_data_per_layer):
    ret = {
        0: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 3, 32, 32)),
            W=np.zeros(shape=(64, 3, 3, 3)),
            B=np.zeros(shape=(64,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        1: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 64, 32, 32)),
            W=np.zeros(shape=(32, 64, 1, 1)),
            B=np.zeros(shape=(32,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(1, 1),
            pads=(0, 0, 0, 0),
            strides=(1, 1),
        ),
        2: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 32, 32, 32)),
            W=np.zeros(shape=(64, 32, 3, 3)),
            B=np.zeros(shape=(64,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        3: make_testnode(
            "ConvReluMaxPool",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 64, 32, 32)),
            W=np.zeros(shape=(32, 64, 3, 3)),
            B=np.zeros(shape=(32,)),
            _maxpool_ceil_mode=0,
            _maxpool_dilations=(1, 1),
            _maxpool_kernel_shape=(2, 2),
            _maxpool_pads=(0, 0, 0, 0),
            _maxpool_strides=(2, 2),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        4: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 32, 16, 16)),
            W=np.zeros(shape=(64, 32, 1, 1)),
            B=np.zeros(shape=(64,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(1, 1),
            pads=(0, 0, 0, 0),
            strides=(1, 1),
        ),
        5: make_testnode(
            "ConvReluMaxPool",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 64, 16, 16)),
            W=np.zeros(shape=(128, 64, 3, 3)),
            B=np.zeros(shape=(128,)),
            _maxpool_ceil_mode=0,
            _maxpool_dilations=(1, 1),
            _maxpool_kernel_shape=(2, 2),
            _maxpool_pads=(0, 0, 0, 0),
            _maxpool_strides=(2, 2),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        6: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 128, 8, 8)),
            W=np.zeros(shape=(128, 128, 1, 1)),
            B=np.zeros(shape=(128,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(1, 1),
            pads=(0, 0, 0, 0),
            strides=(1, 1),
        ),
        7: make_testnode(
            "ConvReluMaxPool",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 128, 8, 8)),
            W=np.zeros(shape=(64, 128, 3, 3)),
            B=np.zeros(shape=(64,)),
            _maxpool_ceil_mode=0,
            _maxpool_dilations=(1, 1),
            _maxpool_kernel_shape=(2, 2),
            _maxpool_pads=(0, 0, 0, 0),
            _maxpool_strides=(2, 2),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        8: make_testnode(
            "ConvRelu",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 64, 4, 4)),
            W=np.zeros(shape=(128, 64, 3, 3)),
            B=np.zeros(shape=(128,)),
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(3, 3),
            pads=(1, 1, 1, 1),
            strides=(1, 1),
        ),
        9: make_testnode(
            "ConvReluMaxPoolReshape",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 128, 4, 4)),
            W=np.zeros(shape=(128, 128, 1, 1)),
            B=np.zeros(shape=(128,)),
            _maxpool_ceil_mode=0,
            _maxpool_dilations=(1, 1),
            _maxpool_kernel_shape=(2, 2),
            _maxpool_pads=(0, 0, 0, 0),
            _maxpool_strides=(2, 2),
            _reshape_allowzero=0,
            _squeeze_factor=[0.25],
            activation=b"relu",
            dilations=(1, 1),
            group=1,
            kernel_shape=(1, 1),
            pads=(0, 0, 0, 0),
            shape=b"/Constant_output_0",
            strides=(1, 1),
        ),
        10: make_testnode(
            "Gemm",
            ["X", "W", "B"],
            ["Y"],
            X=np.zeros(shape=(1, 128, 2, 2)),
            W=np.zeros(shape=(10, 512)),
            B=np.zeros(shape=(10,)),
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
    }
    for layeridx, kernel_data in enumerate(_kernel_data_per_layer):
        idx = "W"  # W (X,W,B)
        w = ret[layeridx].input[idx]
        assert np.prod(w.shape[:-2]) == np.prod(kernel_data.shape[:-2])
        ret[layeridx].input[idx] = kernel_data.reshape(w.shape)
    return ret


cifar10_refnet = make_cifar10_refnet(make_kernel_data_per_layer())

cifar10_ref_row_cols = {
    0: [33, 33],
    1: [31, 31],
    2: [33, 33],
    3: [33, 33],
    4: [15, 15],
    5: [17, 17],
    6: [7, 7],
    7: [9, 9],
    8: [5, 5],
    9: [3, 3],
    10: [0, 0],
}

cifar10_ref_pad = {
    0: [1, 1],
    1: [0, 0],
    2: [1, 1],
    3: [1, 1],
    4: [0, 0],
    5: [1, 1],
    6: [0, 0],
    7: [1, 1],
    8: [1, 1],
    9: [0, 0],
    10: [0, 0],
}

cifar10_ref_inputs = {
    0: [1, 3, 32, 32],
    1: [1, 64, 32, 32],
    2: [1, 32, 32, 32],
    3: [1, 64, 32, 32],
    4: [1, 32, 16, 16],
    5: [1, 64, 16, 16],
    6: [1, 128, 8, 8],
    7: [1, 128, 8, 8],
    8: [1, 64, 4, 4],
    9: [1, 128, 4, 4],
    10: [1, 128, 2, 2],
}

cifar10_ref_outputs = {
    0: [1, 64, 32, 32],
    1: [1, 32, 32, 32],
    2: [1, 64, 32, 32],
    3: [1, 32, 16, 16],
    4: [1, 64, 16, 16],
    5: [1, 128, 8, 8],
    6: [1, 128, 8, 8],
    7: [1, 64, 4, 4],
    8: [1, 128, 4, 4],
    9: [1, 128, 2, 2],
    10: [1, 10, 1, 1],
}


@pytest.mark.parametrize(
    "node, input_shape, expected_outshape, expected_rowcols, expected_pad",
    [
        (
            cifar10_refnet[layeridx],
            cifar10_ref_inputs[layeridx],
            cifar10_ref_outputs[layeridx],
            cifar10_ref_row_cols[layeridx],
            cifar10_ref_pad[layeridx],
        )
        for layeridx in range(len(cifar10_refnet))
    ],
)
def test_row_columns(
    layer, grid, node, input_shape, expected_outshape, expected_rowcols, expected_pad
):
    out_shape = configure_layer_by_node(
        layer,
        grid,
        node,
        input_shape,
    )
    assert out_shape == expected_outshape
    assert [layer.row_count, layer.col_count] == expected_rowcols
    assert [layer.row_pad, layer.col_pad] == expected_pad


def test_row_columns_by_graph_node():
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    typs: set[type] = set([])

    for nodex in model.graph.node:
        try:
            node = Node(nodex, model.graph)

            for name, val in node.input.items():
                typ = type(val)
                typs |= set([typ])
                assert "onnx" not in str(typ)
        except NotImplementedError as nimplex:
            raise nimplex
    assert list(typs) == [np.ndarray, type(None)] or list(typs) == [
        type(None),
        np.ndarray,
    ]


@pytest.mark.skip("only for creating testcases (see above (cifar10_refnet))")
def test_print_tests_():
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10_transformed.onnx")

    nodes = [Node(node, model.graph) for node in model.graph.node]

    node_names_order = [
        "/conv1_1/op/ConvRelu",
        "/conv1_2/op/ConvRelu",
        "/conv1_3/op/ConvRelu",
        "/conv2_1/op/ConvReluMaxPool",
        "/conv2_2/op/ConvRelu",
        "/conv3_1/op/ConvReluMaxPool",
        "/conv3_2/op/ConvRelu",
        "/conv4_1/op/ConvReluMaxPool",
        "/conv4_2/op/ConvRelu",
        "/conv5_1/op/ConvReluMaxPoolReshape",
        "/fc/op/Gemm",
    ]
    found_nodes = [
        node_name in [node.name for node in nodes] for node_name in node_names_order
    ]
    assert all(found_nodes), "did not find" + node_names_order[found_nodes.index(False)]
    nodes_in_order = [
        [node for node in nodes if node.name == node_name][0]
        for node_name in node_names_order
    ]
    for layeridx, node in enumerate(nodes_in_order):
        inps = ["X", "W", "B"]
        inputs = [f"{inp}" for inp in inps]
        outputs = [f"{out}" for out in ["Y"]]
        kwargs = {
            inps[i]: f"np.zeros(shape={val.shape})"
            if val is not None
            else f"np.zeros(shape={cifar10_ref_inputs[layeridx]})"
            for i, (inp, val) in enumerate(node.input.items())
        }
        for attr in node.attributes:
            kwargs[attr] = node.attributes[attr]
        kwargs_str = ",\n        ".join(
            [f"{name}={val}" for name, val in kwargs.items()] + [""]
        )
        print(f"""
    {layeridx}: make_testnode(
        "{node.op_type}",
        {inputs},
        {outputs},
        {kwargs_str}),""")
