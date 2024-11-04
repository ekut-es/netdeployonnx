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

import struct

import numpy as np
import pytest

from .data import cifar10_bias, cifar10_weights


@pytest.mark.parametrize("weights, biases, check_weights, check_biases, dest_file", [
    (
        cifar10_weights.kernels,
        cifar10_bias.correct_bias,
        [
            0x50180000, 0x000002d3, 0xfe08f8f6, 0xfb080715,# noqa: E501
            0x03f20dff, 0xfb05f4f6, 0xecee02f5, 0x09edf403,# noqa: E501
            0x0c06070a, 0x23e115e5, 0xf40114ee, 0x0b18fc27,# noqa: E501
            0xca2316ca, 0xe8f400ff,# noqa: E501
        ],
        [
            [
                0x07, 0xf9, 0xf9, 0x04, 0x07, 0x03, 0xff, 0xfd, 0xf9, 0x01, 0x4a, 0xe7 # noqa: E501
            ],
            [
                #[ruff::skip]
                0xce, 0xc3, 0x7f, 0x7f, 0x7f, 0x1f, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f # noqa: E501
            ],
            [
                #[ruff::skip]
                0x66, 0x03, 0x1f, 0x7f, 0x22, 0x1c, 0x33, 0x47, 0xc9, 0xf0, 0xf9, 0x21 # noqa: E501
            ],
            [
                #[ruff::skip]
                0x49, 0x18, 0x05, 0x40, 0xe1, 0xee, 0xf9, 0xa4, 0x62, 0xf5, 0xe7, 0xf4 # noqa: E501
            ],
        ],
        "cifar10.npy"),
])
def test_create_flash_npy_from_weights_biases(
    weights: list[dict],
    biases: list[list[int]],
    check_weights: list[int],
    check_biases: list[list[int]],
    dest_file: str
    ):
    weights_converted = bytearray()
    for weight_section in weights:
        addr: int = weight_section.get("addr")
        data: bytes = weight_section.get("data")
        weights_converted += struct.pack(">I", addr)
        weights_converted += struct.pack(">I", len(data)//4)
        weights_converted += data


    #print as 4 byte hex array => [0x00000000, 0x00000000, ...]
    # only show the first 10 ints
    ints = 10
    int_entries = [
        int.from_bytes(weights_converted[i:i+4])# big endian
        for i in range(0, len(weights_converted), 4)
        ]
    # print(", ".join([f"0x{u32b:08X}" for u32b in first_entries]))
    assert isinstance(check_weights[0], int), "weights are not integers"
    assert check_weights[:10] == int_entries[:10], "weights are not correct"

    assert len(check_biases) == 4, "biases are not 4 arrays"
    assert len(biases) == 4, "biases are not 4 arrays"

    for idx, check_bias in enumerate(check_biases):
        assert isinstance(check_bias[0], int), "bias is not integer"
        assert check_bias[:10] == biases[idx][:10], "bias on idx {idx} is not correct"

    np_weights = np.array(int_entries)
    np_bias = np.zeros((4, 256))
    for i in range(4):
        np_bias[i, :len(biases[i])] = biases[i]
    weights_n_biases = {
        "weights": np_weights,
        "biases": np_bias,
    }
    np.save(dest_file, weights_n_biases, allow_pickle=True)
    loaded = np.load(dest_file, allow_pickle=True).item()
    assert np.array_equal(loaded.get("weights"), weights_n_biases["weights"])
    assert np.array_equal(loaded.get("biases"), weights_n_biases["biases"])
