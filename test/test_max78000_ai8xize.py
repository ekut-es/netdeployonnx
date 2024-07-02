import asyncio
import logging
from pathlib import Path

import onnx
import pytest

from netdeployonnx.devices.max78000.ai8xize import (
    MAX78000_ai8xize,
)
from netdeployonnx.devices.max78000.core import CNNx16Core

from .data.cifar10_layout import cifar10_layout as cifar10_layout_func

logging.basicConfig(
    level=logging.INFO,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


@pytest.mark.asyncio
async def test_backend_ai8xize():
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    result = await dev.run_onnx(model)
    print(result)
    assert "exception" not in result


@pytest.fixture
def cifar10_layout():
    return asyncio.run(cifar10_layout_func())


@pytest.mark.asyncio
async def test_backend_ai8xize_layout(cifar10_layout):
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    result = await dev.layout_transform(model)
    assert result
    assert isinstance(result, CNNx16Core)
    incorrect = []
    for quad in range(4):
        for layeridx in range(16):
            orig_layer = cifar10_layout[quad, layeridx]
            layer = result[quad, layeridx]
            for fieldname, field in orig_layer.model_fields.items():
                origval = getattr(orig_layer, fieldname)
                val = getattr(layer, fieldname)
                if type(origval) in [int, float, dict, bool]:
                    if origval != val:
                        incorrect.append(
                            f"quad={quad}, layer={layeridx}, field={fieldname}: "
                            f"{origval} != {val}"
                        )
    # from pprint import pprint
    # pprint(incorrect)
    assert len(incorrect) == 0, f"Errors: {len(incorrect)}"
