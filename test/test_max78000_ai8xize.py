import asyncio
import io
import logging
from pathlib import Path
from unittest import mock

import onnx
import pydantic
import pytest
import torch
import torch.nn as nn

from netdeployonnx.devices.max78000.ai8xize import (
    MAX78000_ai8xize,
)
from netdeployonnx.devices.max78000.ai8xize.config import (
    AI8XizeConfig,
    AI8XizeConfigLayer,
)
from netdeployonnx.devices.max78000.core import (
    CNNx16_Layer,
    CNNx16_Processor,
    CNNx16Core,
)

from .data.cifar10_layout import cifar10_layout as cifar10_layout_func

logging.basicConfig(
    level=logging.WARNING,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


c10_layers = [
    AI8XizeConfigLayer(**layer)
    for layer in [
        {
            "out_offset": 0x4000,
            "processors": 0x0000000000000007,  # 1_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "data_format": "HWC",
        },
        {
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 1_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "out_offset": 0x4000,
            "processors": 0x00000000FFFFFFFF,  # 1_3
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 2_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFF00000000,  # 2_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 3_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,  # do 0?
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 3_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 4_1
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 4_2
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            "output_shift": -2,
        },
        {
            "max_pool": 2,
            "pool_stride": 2,
            "out_offset": 0x0000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 5_1
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            "output_shift": -1,
        },
        {
            "flatten": True,
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,
            "operation": "MLP",
            "output_width": 32,
            # "activate": "none",
            "output_shift": 1,
        },
    ]
]


@pytest.fixture
def cifar10_layout():
    return asyncio.run(cifar10_layout_func())


def core_layer_equal(
    original_layer: CNNx16_Layer,
    layer_under_test: CNNx16_Layer,
    quad: int,
    layeridx: int,
) -> list:
    incorrect = []
    # iterate over fields of original layer
    for fieldname, field in original_layer.model_fields.items():
        if fieldname in ["quadrant"]:  # skip these fields
            continue
        # get the values
        origval = getattr(original_layer, fieldname)
        val_under_test = getattr(layer_under_test, fieldname)
        # compare only if the field is a simple type
        if origval != val_under_test:
            incorrect.append(
                f"quad={quad}, layer={layeridx}, field={fieldname}: "
                f"{str(origval)[:30]} != {str(val_under_test)[:30]}"
            )
    return incorrect


def core_processor_equal(
    original_proc: CNNx16_Processor,
    proc_under_test: CNNx16_Processor,
    quad: int,
    proc: int,
) -> list:
    incorrect = []
    for fieldname, field in original_proc.model_fields.items():
        if fieldname in ["quadrant", "layer"]:
            continue  # skip these fields
        origval = getattr(original_proc, fieldname)
        val_under_test = getattr(proc_under_test, fieldname)
        if origval != val_under_test:
            incorrect.append(
                f"quad={quad}, proc={proc}, field={fieldname}: "
                f"{str(origval)[:30]} != {str(val_under_test)[:30]}"
            )
    return incorrect


def core_equal(original_core: CNNx16Core, core_under_test: CNNx16Core) -> bool:
    incorrect = []
    for quad in range(4):
        for fieldname, field in original_core[quad].model_fields.items():
            if fieldname in ["layers", "processors"]:
                continue  # skip these fields
            origval = getattr(original_core[quad], fieldname)
            val_under_test = getattr(core_under_test[quad], fieldname)

            if origval != val_under_test:
                incorrect.append(
                    f"quad={quad}, field={fieldname}: "
                    f"{str(origval)[:30]} != {str(val_under_test)[:30]}"
                )
        # check layers
        for layeridx in range(16):
            incorrect.extend(
                core_layer_equal(
                    original_core[quad, layeridx],
                    core_under_test[quad, layeridx],
                    quad,
                    layeridx,
                )
            )

        # check processors
        for proc in range(16):
            incorrect.extend(
                core_processor_equal(
                    original_core[quad].processors[proc],
                    core_under_test[quad].processors[proc],
                    quad,
                    proc,
                )
            )
    return incorrect


@pytest.mark.asyncio
async def test_backend_ai8xize():
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    result = await dev.run_onnx(model)
    print(result)
    assert "exception" not in result


@pytest.mark.asyncio
async def test_backend_ai8xize_test_execute_cifar10():
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    with mock.patch(
        "netdeployonnx.devices.max78000.device.MAX78000.execute"
    ) as mock_execute:
        await dev.run_onnx(model)
        mock_execute.assert_awaited_once()
        instructions, metrics = mock_execute.await_args.args


@pytest.mark.asyncio
async def test_backend_ai8xize_test_compile_instructions_cifar10(cifar10_layout):
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    with mock.patch(
        "netdeployonnx.devices.max78000.device.MAX78000.compile_instructions"
    ) as mock_compile_instructions:
        mock_compile_instructions.return_value = [
            {"stage": "cnn_enable", "instructions": []}
        ]
        await dev.run_onnx(model)
        mock_compile_instructions.assert_awaited_once()
        # unpack the args
        (layout_ir,) = mock_compile_instructions.await_args.args
        assert isinstance(layout_ir, CNNx16Core)

        core_equal_result = core_equal(cifar10_layout, layout_ir)
        assert core_equal_result == [], f"layout mismatch (len={len(core_equal_result)})"


@pytest.mark.asyncio
async def test_backend_ai8xize_layout(cifar10_layout):
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    result = await dev.layout_transform(model)
    assert result
    assert isinstance(result, CNNx16Core)

    core_equal_result = core_equal(cifar10_layout, result)
    assert core_equal_result == [], f"layout mismatch (len={len(core_equal_result)})"


def test_ai8xconfig():
    cfg = AI8XizeConfig(
        arch="ai85nascifarnet",
        dataset="CIFAR10",
        layers=[
            AI8XizeConfigLayer(
                **{
                    "out_offset": 0x4000,
                    "processors": 0x0000000000000007,  # 1_1
                    "operation": "conv2d",
                    "kernel_size": "3x3",
                    "pad": 1,
                    "activate": "ReLU",
                    "data_format": "HWC",
                }
            ),
        ],
    )

    d = cfg.model_dump(exclude_defaults=True)
    assert d.get("arch") == "ai85nascifarnet"
    assert d.get("dataset") == "CIFAR10"
    assert len(d.keys()) == 3
    assert len(d.get("layers")) == 1
    layer0 = d.get("layers")[0]
    assert layer0.get("out_offset") == 0x4000
    assert layer0.get("processors") == 7

    # from pprint import pprint
    # pprint(d)


def test_ai8xconfig_missing_attribute():
    with pytest.raises(pydantic.ValidationError):
        AI8XizeConfig(
            arch="ai85nascifarnet",
            dataset="CIFAR10",
            layers=[
                AI8XizeConfigLayer(**{}),
            ],
        )


@pytest.mark.asyncio
async def test_layout_transform_load():
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    max78000 = MAX78000_ai8xize()
    result = await max78000.layout_transform(model)
    assert result


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


@pytest.mark.asyncio
async def test_layout_transform_simple_model():
    model = SimpleModel()
    bytesio = io.BytesIO()
    dummy_input = torch.tensor([1.0]).unsqueeze(0).cpu()
    torch.onnx.export(
        model,
        dummy_input,
        bytesio,
    )
    bytesio.seek(0)
    onnx_model = onnx.load(bytesio)
    onnx.checker.check_model(onnx_model)
    max78000 = MAX78000_ai8xize()
    result = await max78000.layout_transform(onnx_model)
    assert result


def test_layout_transform_generate_config_from_model():
    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")

    default_values = {
        field_name: field.default
        for field_name, field in AI8XizeConfigLayer.model_fields.items()
    }

    max78000 = MAX78000_ai8xize()
    result, input_shape = max78000.generate_config_from_model(model)
    assert result
    layers = result.pop("layers")
    for layeridx, layer in enumerate(c10_layers):
        c10_layerdict = layer.model_dump(exclude_unset=True)
        if "name" in layers[layeridx]:
            layers[layeridx].pop("name")

        # check for missing keys
        assert len(set(c10_layerdict.keys()) - set(layers[layeridx].keys())) == 0, (
            f"missing keys ({set(c10_layerdict.keys())- set(layers[layeridx].keys())})"
            f" in Layer {layeridx}"
        )

        # check for extra keys, but ignore extra keys that are default values
        for extra_key in set(layers[layeridx].keys()) - set(c10_layerdict.keys()):
            # check if the extra key is a default value
            assert (
                layers[layeridx][extra_key] == default_values[extra_key]
            ), f"unexpected value in Layer {layeridx} for key {extra_key}"
            # it is a default value, so remove it from the dict
            layers[layeridx].pop(extra_key)

        # check for different values
        assert (
            c10_layerdict == layers[layeridx]
        ), f"different values in Layer {layeridx}"
