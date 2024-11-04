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
import pytest

from netdeployonnx.devices.max78000 import MAX78000
from netdeployonnx.devices.max78000.core import CNNx16Core


@pytest.fixture(scope="module")
def device():
    return MAX78000()


@pytest.mark.asyncio
async def test_device_compile_instructions_none(device):
    compiled = await device.compile_instructions(None)
    assert compiled == [
        {"stage": "cnn_enable", "instructions": []},
        {"stage": "cnn_init", "instructions": []},
        {"stage": "cnn_load_weights", "instructions": []},
        {"stage": "cnn_load_bias", "instructions": []},
        {"stage": "cnn_configure", "instructions": []},
        {"stage": "load_input", "instructions": []},
        {"stage": "cnn_start", "instructions": []},
        {"stage": "done", "instructions": []},
    ]


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.asyncio
async def test_device_compile_instructions(device):
    compiled = await device.compile_instructions(CNNx16Core())
    stages = {
        "cnn_enable": [],
        "cnn_init": [],
        "cnn_load_weights": [],
        "cnn_load_bias": [],
        "cnn_configure": [],
        "load_input": [],
        "cnn_start": [],
        "done": [],
    }
    for stage, instructions in stages.items():
        # stage looks like {"stage": "X", "instructions": []}
        assert stage in [s["stage"] for s in compiled]
        compiled_stage = [s for s in compiled if s["stage"] == stage]
        assert compiled_stage and len(compiled_stage) == 1
        compiled_stage = compiled_stage[0]
        assert compiled_stage["instructions"] == instructions
