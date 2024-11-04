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
import numpy as np

from netdeployonnx.devices.max78000.core import MAX_PROCESSORS, CNNx16Core
from netdeployonnx.devices.max78000.synthesizer.grid import Node


class KernelReference:
    def __init__(self, ref: str): ...

    def __len__(self):
        "virtual length"
        return self.size


class PerChanProcessorMRAM:
    def __init__(self):
        self.kernels: dict[int, KernelReference] = {}


class MemoryQuadrant:
    def __init__(
        self,
        processors: int = 0,
        per_layer_quantization: list[int] = [],
        tensor: ... = None,
    ):
        """
        Memory quadrant for the MAX78000 device

        processors: int
            Number of processors in the quadrant
        """
        self.processors = processors
        self.data = []

    def _assert_valid(self):
        return True

    def get_kernel_data(self, inch: int, outch: int, packed72bit: bool = True) -> bytes:
        """
        We only do CHW format for now =>
            CCC CCC CCC CCC
            CCC CCC CCC CCC
            CCC CCC CCC CCC
            CCC CCC CCC CCC
            CCC CCC CCC CCC
            => 3 X 5 X 4


        packed72bit: bool
            pack to 72 bits (mexpress), if False, we have to expand to 3 words
            per kernel and leave it as 0 => [0,0,0,k[0], k[1], ...., k[8],]
                                 instead of [k[0], k[1], ...., k[8]]
        """
        assert self._assert_valid()
        consolidated_memory = self.data[0]

        if consolidated_memory.shape[-2:] == (3, 3):
            # we have a 3x3 kernel
            # we need to reshape it to 9x1
            memslots = consolidated_memory.reshape(-1, 9)  # 9 byte in 72bits
            data = np.zeros_like(memslots, dtype=np.uint8)
            idx = 0
            assert outch * inch == consolidated_memory.shape[0], "have to be same shape"

            # reorder memslots so it works for our processor
            for cidx in self.kernel_reorder_generator(inch, outch):
                # check if we even have this data
                if memslots.shape[0] > cidx and data.shape[0] > idx:
                    # we do have the data, so reorder
                    data[idx] = memslots[cidx]
                    idx += 1
                else:
                    raise Exception(
                        f"no find: {cidx} in shape[0]={memslots.shape[0]}"
                        f" or {idx} in shape[0]={data.shape[0]}"
                    )
        elif consolidated_memory.shape[-2:] == (1, 1):
            # we have a 1x1 kernel
            data = consolidated_memory.view(dtype=np.uint8)
            # no reordering needed
        return data.reshape(-1).tobytes()

    def kernel_reorder_generator(self, tot_in_chs: int, tot_out_chs: int) -> list[int]:
        for in_ch in range(tot_in_chs):
            for out_ch in range(tot_out_chs):
                # we would do
                # src_offs = ch + m * in_ch
                # but to stay in order:
                yield in_ch + out_ch * tot_in_chs


class MemoryPackager: ...


def layout_mem(core: CNNx16Core, layeridx: int, node: Node) -> MemoryQuadrant:
    in_shape = node.input["X"].shape
    mul_shape = node.input["W"].shape
    bias_shape = node.input["B"].shape  # noqa: F841

    input_channels = in_shape[1]
    output_channels = mul_shape[0]
    # check if we need to expand?
    expansion_coeff = output_channels / input_channels
    expansion_mode = input_channels > MAX_PROCESSORS
    multipass = expansion_coeff > 1
    print("ok", expansion_coeff, expansion_mode, multipass)

    for input_channelid in range(input_channels):
        procid = input_channelid % MAX_PROCESSORS
        quad = CNNx16Core.processor_quadrant(procid)
        core[quad].add_quadkernel_for_layer(
            layeridx=layeridx,
            processor_id=procid,  # we have max 64 channels per core
            kernel_data=node.input["W"][:, input_channelid, :, :],
        )
