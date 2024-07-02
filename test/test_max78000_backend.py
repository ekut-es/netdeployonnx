# test_max78000_graph_synthesis.py
import logging
from unittest import mock

import numpy as np

from netdeployonnx.devices.max78000.synthesizer.backend import (
    configure_layer_by_node,
    load_backends,
)

logging.basicConfig(
    level=logging.INFO,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


def test_load_backends():
    # TODO: finish tests
    load_backends()


def test_configure_layer_by_node():
    # TODO: finish tests
    layer = mock.MagicMock()
    grid = mock.MagicMock()
    node = mock.MagicMock()
    node.op_type = "Gemm"
    node.input = {i: np.zeros(shape=(1, 1, 1, 1)) for i in range(3)}
    out_shape = configure_layer_by_node(
        layer,
        grid,
        node,
        [1, 1, 1, 1],
    )
    assert out_shape == [1, 1, 1, 1]
