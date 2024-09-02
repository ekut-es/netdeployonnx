import asyncio
import io
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pydantic
import pytest
import torch
import torch.nn as nn

from netdeployonnx.devices.max78000 import MAX78000Metrics
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
from .test_serialhandler import (
    MeasureDevice,
    open_serial_connection_virtual_device,  # noqa: F401
    print_chunks,
)

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


@pytest.fixture
def test_instructions():
    return [
        {"stage": "cnn_enable", "instructions": [("ACTION", 20, 0)]},
        {
            "stage": "cnn_init",
            "instructions": [
                ("CNNx16_AOD_CTRL", 0),
                ("CNNx16_0_CTRL", 1048584),
                ("CNNx16_0_SRAM", 1036),
                ("CNNx16_0_LCNT_MAX", 15),
                ("CNNx16_1_CTRL", 1048584),
                ("CNNx16_1_SRAM", 1036),
                ("CNNx16_1_LCNT_MAX", 15),
                ("CNNx16_2_CTRL", 1048584),
                ("CNNx16_2_SRAM", 1036),
                ("CNNx16_2_LCNT_MAX", 15),
                ("CNNx16_3_CTRL", 1048584),
                ("CNNx16_3_SRAM", 1036),
                ("CNNx16_3_LCNT_MAX", 15),
                "",
            ],
        },
        {"stage": "cnn_load_weights", "instructions": [("ACTION", 40, 0)]},
        {
            "stage": "cnn_load_bias",
            "instructions": [
                (
                    1343258624,
                    b"\x07\xf9\xf9\x04\x07\x03\xff\xfd\xf9\x01I\xe7\x1d4R^47\xef.t\xfc",
                ),
                (
                    1347452928,
                    b"\xce\xc3~~~\x1e\x80~\x80\x80~~\x82~\x80\x123~~\xdf~\x80\x80\xda~~",
                ),
                (
                    1351647232,
                    b"e\x02\x1e~!\x1b2F\xc9\xf0\xf9 $\xec!~3\x05\xe6\xfc3\xe7\x1d\xdb",
                ),
                (
                    1355841536,
                    b"H\x17\x04?\xe1\xee\xf9\xa4a\xf5\xe7\xf4/\x1c\x05\x07\xcc\xf5\x11",
                ),
            ],
        },
        {
            "stage": "cnn_configure",
            "instructions": [
                "// Layer 0 quadrant 0",
                ("CNNx16_0_L0_RCNT", 65569),
                ("CNNx16_0_L0_CCNT", 65569),
                ("CNNx16_0_L0_WPTR_BASE", 4096),
                ("CNNx16_0_L0_WPTR_MOFF", 8192),
                ("CNNx16_0_L0_LCTRL0", 11040),
                ("CNNx16_0_L0_MCNT", 504),
                ("CNNx16_0_L0_TPTR", 31),
                ("CNNx16_0_L0_EN", 458759),
                ("CNNx16_0_L0_LCTRL1", 129024),
                "",
                "// Layer 0 quadrant 1",
                ("CNNx16_1_L0_RCNT", 65569),
                ("CNNx16_1_L0_CCNT", 65569),
                ("CNNx16_1_L0_WPTR_BASE", 4096),
                ("CNNx16_1_L0_WPTR_MOFF", 8192),
                ("CNNx16_1_L0_LCTRL0", 2848),
                ("CNNx16_1_L0_MCNT", 504),
                ("CNNx16_1_L0_TPTR", 31),
                ("CNNx16_1_L0_POST", 4224),
                ("CNNx16_1_L0_LCTRL1", 129024),
                "",
                "// Layer 0 quadrant 2",
                ("CNNx16_2_L0_RCNT", 65569),
                ("CNNx16_2_L0_CCNT", 65569),
                ("CNNx16_2_L0_WPTR_BASE", 4096),
                ("CNNx16_2_L0_WPTR_MOFF", 8192),
                ("CNNx16_2_L0_LCTRL0", 2848),
                ("CNNx16_2_L0_MCNT", 504),
                ("CNNx16_2_L0_TPTR", 31),
                ("CNNx16_2_L0_LCTRL1", 129024),
                "",
                "// Layer 0 quadrant 3",
                ("CNNx16_3_L0_RCNT", 65569),
                ("CNNx16_3_L0_CCNT", 65569),
                ("CNNx16_3_L0_WPTR_BASE", 4096),
                ("CNNx16_3_L0_WPTR_MOFF", 8192),
                ("CNNx16_3_L0_LCTRL0", 2848),
                ("CNNx16_3_L0_MCNT", 504),
                ("CNNx16_3_L0_TPTR", 31),
                ("CNNx16_3_L0_LCTRL1", 129024),
                "",
            ],
        },
        {"stage": "load_input", "instructions": []},
        {
            "stage": "cnn_start",
            "instructions": [
                ("CNNx16_0_CTRL", 1050632),
                ("CNNx16_1_CTRL", 1050633),
                ("CNNx16_2_CTRL", 1050633),
                ("CNNx16_3_CTRL", 1050633),
                "",
                ("CNNx16_0_CTRL", 1048585),
            ],
        },
        {"stage": "done", "instructions": []},
    ]


def check_off_by_one_error(
    origval: bytes, val_under_test: bytes, quad: int, fieldname: str
) -> list:
    # we should calculate a diff vector, because onnxcp and checkpoint
    # uses a different method to quantize
    #
    # w = np.floor((checkpoint_state[bias_name] / 2**(wb - 1))
    # .numpy()).astype(np.int64)
    incorrect = []
    ori_int = np.frombuffer(origval, dtype=np.int8)
    vut_int = np.frombuffer(val_under_test, dtype=np.int8)
    diff_vector = ori_int - vut_int
    # is the difference max 1?
    if np.abs(diff_vector).max() > 1:
        # yes it is
        incorrect.append(
            f"quad={quad}, field={fieldname}: "
            f"{str(origval)[:30]} != {str(val_under_test)[:30]}"
        )
        print_chunks(ori_int, vut_int, diff_vector)
    return incorrect


def core_layer_equal(
    original_layer: CNNx16_Layer,
    layer_under_test: CNNx16_Layer,
    quad: int,
    layeridx: int,
) -> list:
    """compare the fields of the layer, mostly idx,
    layer_field_dict, row_count, row pad, ...
    """
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
    "compare the fields of the processor, mostly idx, enabled, kernels"
    incorrect = []
    for fieldname, field in original_proc.model_fields.items():
        if fieldname in ["quadrant", "layer"]:
            continue  # skip these fields
        origval = getattr(original_proc, fieldname)
        val_under_test = getattr(proc_under_test, fieldname)
        if origval != val_under_test:
            if fieldname == "kernels":
                kernel_lengths_orig = {k: len(v) for k, v in origval.items()}
                kernel_lengths_vut = {k: len(v) for k, v in val_under_test.items()}
                assert (
                    kernel_lengths_orig == kernel_lengths_vut
                ), "kernel lengths differ"
                for k in origval:
                    orig_kernel = origval[k]
                    vut_kernel = val_under_test[k]
                    incorrect.extend(
                        check_off_by_one_error(orig_kernel, vut_kernel, quad, fieldname)
                    )
            else:
                incorrect.append(
                    f"quad={quad}, proc={proc}, field={fieldname}: "
                    f"{str(origval)[:30]} != {str(val_under_test)[:30]}"
                )
    return incorrect


def core_equal(original_core: CNNx16Core, core_under_test: CNNx16Core) -> bool:  # noqa: C901
    "compare the fields of the core, mostly idx, bias, input_frame_size, mlat_data"
    incorrect = []
    for quad in range(4):
        for fieldname, field in original_core[quad].model_fields.items():
            if fieldname in ["layers", "processors"]:
                continue  # skip these fields
            origval = getattr(original_core[quad], fieldname)
            val_under_test = getattr(core_under_test[quad], fieldname)

            if origval != val_under_test:
                if fieldname == "bias":  # check for off by one error
                    incorrect.extend(
                        check_off_by_one_error(origval, val_under_test, quad, fieldname)
                    )
                    if len(incorrect) > 10:
                        assert False, f"too many errors: {len(incorrect)}, {incorrect}"
                else:
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


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "5.20414e-07,7.38719e-06,0.00310935,0.0735575,2.19814e-06,2.77844e-05,0.00310935,0.0822235,1.92158e-06,2.76521e-05,0.00310935,0.0726008",
            {},
        ),
        (
            "6.21383e-07,7.81052e-06,0.00310935,0.0826666,2.13321e-06,2.77854e-05,0.00310935,0.0798838,2.05103e-06,2.76256e-05,0.00310935,0.0773532",
            {},
        ),
        (
            "5.86322e-07,7.62531e-06,0.00310935,0.0800009,2.19136e-06,2.7759e-05,0.00310935,0.082052,2.21465e-06,2.76521e-05,0.00310935,0.0831993",
            {},
        ),
        (
            "8.76824e-07,7.73115e-06,0.00310935,0.116524,1.90149e-06,2.7759e-05,0.00310935,0.0716095,2.29047e-06,2.7705e-05,0.00310935,0.085783",
            {},
        ),
        (
            "8.04126e-07,7.73115e-06,0.00310935,0.107121,1.99366e-06,2.77844e-05,0.00310935,0.0748642,2.25008e-06,2.76785e-05,0.00310935,0.0844028",
            {},
        ),
        (
            "7.43405e-07,7.7576e-06,0.00310935,0.0989386,2.19818e-06,2.77579e-05,0.00310935,0.0823006,2.21161e-06,2.76785e-05,0.00310935,0.0830128",
            {},
        ),
        (
            "7.49766e-07,7.70469e-06,0.00310935,0.100422,2.2807e-06,2.7759e-05,0.00310935,0.0852701,2.14223e-06,2.75992e-05,0.00310935,0.0807286",
            {},
        ),
        (
            "4.72946e-07,7.83698e-06,0.00310935,0.0634573,2.27616e-06,2.77844e-05,0.00310935,0.0850316,2.16434e-06,2.76785e-05,0.00310935,0.0813048",
            {},
        ),
        (
            "4.88699e-07,7.5724e-06,0.00310935,0.0676462,2.27405e-06,2.7759e-05,0.00310935,0.0850307,2.19117e-06,2.76521e-05,0.00310935,0.0823499",
            {},
        ),
        (
            "5.90711e-07,7.96927e-06,0.00310935,0.077233,2.19803e-06,2.7759e-05,0.00310935,0.0822919,1.9502e-06,2.76521e-05,0.00310935,0.0736356",
            {},
        ),
        (
            "6.39399e-07,7.81052e-06,0.00310935,0.0849732,2.15581e-06,2.7759e-05,0.00310935,0.0807713,2.14708e-06,2.7705e-05,0.00310935,0.0806073",
            {
                "uJ_per_weights_loading": 0.639,
                "us_per_weights_loading": 7.81,
                "uW_per_weights_loading": 81863.85,
                "uJ_per_input_loading": 2.16,
                "us_per_input_loading": 27.75,
                "uW_per_input_loading": 77662.0,
                "uJ_per_convolution": 0,
                "us_per_convolution": 0,
                "uW_per_convolution": 0,
            },
        ),
        (
            "7.50678e-07,7.73115e-06,0.00310935,0.100207,2.1794e-06,2.7759e-05,0.00310935,0.0816211,2.26539e-06,2.78373e-05,0.00310935,0.084489",
            {
                "uJ_per_weights_loading": 0.75,
                "us_per_weights_loading": 7.73,
                "uW_per_weights_loading": 97090.0,
                "uJ_per_input_loading": 2.18,
                "us_per_input_loading": 27.75,
                "uW_per_input_loading": 78510.0,
                "uJ_per_convolution": 0.09,
                "us_per_convolution": 0.08,
                "uW_per_convolution": 2867.9,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_metrics_parsing(message, expected):
    mdev = MeasureDevice()

    async def return_virt(*args, **kwargs):
        reader, writer = mock.AsyncMock(), mock.AsyncMock()
        reader.read = mdev.read
        writer.drain = mdev.drain
        writer.write = mdev.write
        writer.close = mock.MagicMock()
        reader.close = mock.MagicMock()
        return reader, writer

    async def my_read(inst):
        return bytes(f"{message}\r\n", "utf-8")

    mdev.read = my_read

    with mock.patch(
        "serial_asyncio.open_serial_connection", return_virt
    ) as mock_open_serial_connection:  # noqa: F841
        metrics = MAX78000Metrics("/dev/null")
        await metrics.set_mode("triggered")
        await metrics.collect()
        data = metrics.as_dict()

        assertion_errors = []
        # assert len(data) == len(
        #     expected
        # ), f"different keys: {list(data.keys())} != {list(expected.keys())}"
        for key in expected:
            try:
                if isinstance(expected[key], float):
                    r, a = estimate_isclose_tolerances(data[key], expected[key])
                    assert np.isclose(data[key], expected[key], rtol=1e-3, atol=1e-3), (
                        f"{key}: {data[key]} != {expected[key]},"
                        f" rtol={r:g}, atol={a:g}"
                    )
                else:
                    assert data[key] == expected[key]
            except AssertionError as e:
                assertion_errors.append(str(e).split("\n")[0])
        if len(assertion_errors) > 0:
            for e in assertion_errors:
                print(e)
            assert False, "\n".join([str(e) for e in assertion_errors])


@pytest.mark.parametrize(
    "measurement, expected",
    [
        (
            {
                "idle_power": [10.87] * 3,  # in mW
                "active_power": [70.3, 69.5, 327.8],  # in mW
                "time": [20.8e-3, 268.3e-6, 1.6e-3],  # in s
                "power": [0.1, 0.2, 0.3, 0.4],  # in W
                "voltages": [3.3, 3.3, 3.3, 1.8],  # in V
            },
            {
                "deployment_execution_times": {"total": 0.0},
                # these are calculated from above
                "uJ_per_all": 1742.73,
                "uJ_per_convolution": 491.0,
                "uJ_per_input_loading": 15.73,
                "uJ_per_weights_loading": 1236.0,
                # these are absolute (from above)
                "uW_per_all": 376.1e3,
                "uW_per_convolution": 258_300.0,
                "uW_per_input_loading": 69500.0 - 10870,  # TODO: why 10870?
                "uW_per_weights_loading": 70300.0 - 10870,  # TODO: why 10870?
                "us_per_all": 22.4e3,
                "us_per_convolution": 1.3317e3,
                "us_per_input_loading": 268.3,
                "us_per_weights_loading": 20.8e3,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_metrics_dict_with_virtual_serialport(measurement, expected):
    mdev = MeasureDevice(measurement)

    async def return_virt(*args, **kwargs):
        reader, writer = mock.AsyncMock(), mock.AsyncMock()
        reader.read = mdev.read
        writer.drain = mdev.drain
        writer.write = mdev.write
        writer.close = mock.MagicMock()
        reader.close = mock.MagicMock()
        return reader, writer

    with mock.patch(
        "serial_asyncio.open_serial_connection", return_virt
    ) as mock_open_serial_connection:  # noqa: F841
        metrics = MAX78000Metrics("/dev/null")
        await metrics.set_mode("triggered")
        await metrics.collect()
        data = metrics.as_dict()

        assertion_errors = []
        for key in expected:
            try:
                if isinstance(expected[key], float):
                    r, a = estimate_isclose_tolerances(data[key], expected[key])
                    assert np.isclose(data[key], expected[key], rtol=1e-3, atol=1e-3), (
                        f"{key}: {data[key]} != {expected[key]},"
                        f" rtol={r:g}, atol={a:g}"
                    )
                else:
                    assert data[key] == expected[key]
            except AssertionError as e:
                assertion_errors.append(str(e).split("\n")[0])
        if len(assertion_errors) > 0:
            for e in assertion_errors:
                print(e)
            assert False, "\n".join([str(e) for e in assertion_errors])


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("power", "0.0001,0.0002,0.0003,0.0004\r\n"),
        ("voltage", "3.3,3.3,3.3,1.8\r\n"),
        ("current", "0.030303,0.0606061,0.0909091,0.222222\r\n"),
        (
            "triggered",
            "0.00146162,0.0208,3e-05,0.0703,1.86388e-05,0.0002683,"
            "3e-05,0.0695,0.000524432,0.0016,3e-05,0.3278\r\n",
        ),
        (
            "system",
            "6.24e-07,0.0208,3e-05,8.049e-09,0.0002683,3e-05,4.8e-08,0.0016,3e-05\r\n",
        ),
    ],
)
@pytest.mark.asyncio
async def test_metrics_collect_with_virtual_serialport(
    open_serial_connection_virtual_device, mode, expected
):
    with mock.patch(
        "serial_asyncio.open_serial_connection", open_serial_connection_virtual_device
    ) as mock_open_serial_connection:  # noqa: F841
        metrics = MAX78000Metrics("/dev/null")
        await metrics.set_mode(mode)
        data = await metrics.collect()
        assert data == expected


def estimate_isclose_tolerances(a, b):
    atol = abs(a - b)
    rtol = atol / max(abs(a), abs(b))
    return rtol, atol


@pytest.mark.asyncio
async def test_backend_ai8xize_execute_called_cifar10():
    dev = MAX78000_ai8xize()

    data_folder = Path(__file__).parent / "data"
    model = onnx.load(data_folder / "cifar10.onnx")
    with mock.patch(
        "netdeployonnx.devices.max78000.device.MAX78000.execute"
    ) as mock_execute:
        await dev.run_onnx(model, None)
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
        await dev.run_onnx(model, None)
        mock_compile_instructions.assert_awaited_once()
        # unpack the args
        (layout_ir,) = mock_compile_instructions.await_args.args
        assert isinstance(layout_ir, CNNx16Core)

        core_equal_result = core_equal(cifar10_layout, layout_ir)
        assert (
            core_equal_result == []
        ), f"layout mismatch (len={len(core_equal_result)})"  # noqa: E501


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
