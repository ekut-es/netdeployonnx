#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
