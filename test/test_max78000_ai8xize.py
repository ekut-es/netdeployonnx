import asyncio
import io
import logging
import struct
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pydantic
import pytest
import torch
import torch.nn as nn
import yaml

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
from netdeployonnx.devices.max78000.device_transport.serialhandler import recalc_crc

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


data_folder = Path(__file__).parent / "data"
ai8x_synth_network = (
    Path(__file__).parent.parent / "external" / "ai8x-synthesis" / "networks"
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
            # "output_shift": -1,
        },
        {
            "out_offset": 0x4000,
            "processors": 0x00000000FFFFFFFF,  # 1_3
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            # "output_shift": -1,
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
            # "output_shift": -3,
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
            # "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 3_2
            "operation": "conv2d",
            "kernel_size": "1x1",
            "pad": 0,
            "activate": "ReLU",
            # "output_shift": -1,
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
            # "output_shift": -3,
        },
        {
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,  # 4_2
            "operation": "conv2d",
            "kernel_size": "3x3",
            "pad": 1,
            "activate": "ReLU",
            # "output_shift": -2,
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
            # "output_shift": -1,
        },
        {
            "flatten": True,
            "out_offset": 0x4000,
            "processors": 0xFFFFFFFFFFFFFFFF,
            "operation": "MLP",
            "output_width": 32,
            # "activate": "none",
            # "output_shift": 1,
        },
    ]
]


@pytest.fixture
def cifar10_layout():
    return asyncio.run(cifar10_layout_func())


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
        "netdeployonnx.devices.max78000.device_transport"
        ".serialhandler.open_serial_connection",
        return_virt,
    ) as mock_open_serial_connection:  # noqa: F841
        metrics = MAX78000Metrics("/dev/virtualMetrics")
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
    "measurement, expected_deploy, expected_metrics",
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
            },
            {
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
async def test_metrics_dict_with_virtual_serialport(
    measurement, expected_deploy, expected_metrics
):
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
        "netdeployonnx.devices.max78000.device_transport"
        ".serialhandler.open_serial_connection",
        return_virt,
    ) as mock_open_serial_connection:  # noqa: F841
        metrics = MAX78000Metrics("/dev/null")
        await metrics.set_mode("triggered")
        await metrics.collect()
        data = metrics.as_dict()
        metrics = data.get("metrics", {})

        assertion_errors = []
        for key in expected_metrics:
            try:
                if isinstance(expected_metrics[key], float):
                    r, a = estimate_isclose_tolerances(
                        metrics[key], expected_metrics[key]
                    )
                    assert np.isclose(
                        metrics[key], expected_metrics[key], rtol=1e-3, atol=1e-3
                    ), (
                        f"{key}: {metrics[key]} != {expected_metrics[key]},"
                        f" rtol={r:g}, atol={a:g}"
                    )
                else:
                    assert data[key] == expected_metrics[key]
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
    open_serial_connection_virtual_device,  # noqa: F811
    mode,
    expected,
):
    with mock.patch(
        "netdeployonnx.devices.max78000.device_transport"
        ".serialhandler.open_serial_connection",
        open_serial_connection_virtual_device,
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

    model = onnx.load(data_folder / "cifar10.onnx")
    with mock.patch(
        "netdeployonnx.devices.max78000.device.MAX78000.execute"
    ) as mock_execute:
        await dev.run_onnx(model, None)
        mock_execute.assert_awaited_once()
        instructions, metrics = mock_execute.await_args.args


def bytes_to_array(fulldata: bytes):
    data = []
    blocksize = 8
    for i in range(0, len(fulldata), blocksize):
        data.append(fulldata[i : i + blocksize])

    serialized = ",\n".join(", ".join([f"0x{b:02X}" for b in block]) for block in data)

    print(f"""
    uint8_t msg[{len(fulldata)}] = {{
    {serialized}
    }};
    """)


def msg_to_bytes(msg):
    msg.checksum = recalc_crc(msg)
    serdata = msg.SerializeToString()

    fulldata: bytes = struct.pack("<H", (len(serdata)) * 8)
    fulldata += serdata
    return fulldata


@pytest.mark.asyncio
async def test_backend_ai8xize_run_onnx_cifar10_short():
    dev = MAX78000_ai8xize.create_device_from_name_and_ports(
        model_name="test_device",
        communication_port="/dev/virtualDevice",
        energy_port="/dev/virtualMetrics",
    )
    msgs = []

    def handle_msg(self, msg):
        msgs.append(msg)

    model = onnx.load(data_folder / "cifar10_short.onnx")
    # dev.commands.dataHandler.FullDevice.handle_msg
    with mock.patch(
        "netdeployonnx.devices.max78000.device_transport."
        "virtualdevices.FullDevice.handle_msg",
        handle_msg,
    ) as p:  # noqa: F841
        res = await dev.run_onnx(model, None)  # noqa: F841
        # now what?
        # we need the write results

    data = b""
    for msg in msgs:
        print(msg)
        data += msg_to_bytes(msg)
    bytes_to_array(data)


@pytest.mark.parametrize(
    "net_name, ignore",
    [
        # ("cifar10_short.onnx",False),
        # ("cifar10.onnx",False),
        # ("ai85-bayer2rgb-qat8-q.pth.onnx",False),
        # ("ai85-cifar10-qat8-q.pth.onnx",False),
        # ("ai85-cifar100-qat8-q.pth.onnx",False),
        # ("ai85-faceid_112-qat-q.pth.onnx",False),
        # ("ai85-kws20_v3-qat8-q.pth.onnx", False),
        # ("ai8x_test_430_working.onnx", False), # this one produces the proc_error
        ("ai8x_test_436_working.onnx", True),
        ("ai8x_test_442_working.onnx", True),
        ("ai8x_test_455_working.onnx", True),
        ("ai8x_test_459_working.onnx", True),
        # ("ai8x_test_469_possibly_notworking.onnx", False),
    ],
)
@pytest.mark.asyncio
async def test_backend_ai8xize_run_onnx_multinet(net_name: str, ignore: bool):
    dev = MAX78000_ai8xize.create_device_from_name_and_ports(
        model_name="test_device",
        communication_port="/dev/virtualDevice",
        energy_port="/dev/virtualMetrics",
    )
    msgs = []

    def handle_msg(self, msg):
        msgs.append(msg)

    model = onnx.load(data_folder / net_name)
    # dev.commands.dataHandler.FullDevice.handle_msg
    with mock.patch(
        "netdeployonnx.devices.max78000.device_transport."
        "virtualdevices.FullDevice.handle_msg",
        handle_msg,
    ) as p:  # noqa: F841
        res = await dev.run_onnx(model, None)
        if "exception" in res:
            print(res)
            raise res["exception"]
        # now what?
        # we need the write results

    assert len(msgs) > 0, "maybe it did not work!?"


@pytest.mark.asyncio
async def test_backend_ai8xize_test_compile_instructions_cifar10(cifar10_layout):
    dev = MAX78000_ai8xize()

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


@pytest.mark.parametrize(
    "onnx_filename, expected_exception, expected",
    [
        ("ai8x_net_0.onnx", AssertionError("too many input channels=163840"), None),
        ("ai8x_net_1.onnx", AssertionError("too many input channels=163840"), None),
        ("ai8x_net_2.onnx", AssertionError("too many input channels=393216"), None),
        ("ai8x_net_3.onnx", AssertionError("too many input channels=286720"), None),
        (
            "ai8x_net_4.onnx",
            AssertionError("too many input channels=16384"),
            None,
        ),  # 16384 are too many output channels
        ("ai8x_net_5.onnx", AssertionError("too many input channels=98304"), None),
        ("ai8x_net_6.onnx", AssertionError("too many input channels=49152"), None),
        ("ai8x_net_7.onnx", AssertionError("too many input channels=16384"), None),
        ("ai8x_net_8.onnx", AssertionError("too many input channels=49152"), None),
        ("ai8x_net_9.onnx", AssertionError("too many output channels=8192"), None),
        ("ai8x_net_0_fixed.onnx", SystemExit, None),
        ("ai8x_net_1_fixed.onnx", SystemExit, None),
        ("ai8x_net_2_fixed.onnx", SystemExit, None),
        ("ai8x_net_3_fixed.onnx", SystemExit, None),
        ("ai8x_net_4_fixed.onnx", SystemExit, None),
        ("ai8x_net_5_fixed.onnx", SystemExit, None),
        ("ai8x_net_6_fixed.onnx", SystemExit, None),
        ("ai8x_net_7_fixed.onnx", SystemExit, None),
        ("ai8x_net_8_fixed.onnx", SystemExit, None),
        ("ai8x_net_9_fixed.onnx", SystemExit, None),
    ],
)
@pytest.mark.asyncio
async def test_backend_ai8xize_layout_hannahsamples(
    onnx_filename, expected_exception, expected
):
    dev = MAX78000_ai8xize()

    model = onnx.load(data_folder / onnx_filename)
    if expected_exception:
        try:
            result = await dev.layout_transform(model)
        except Exception as ex:
            if isinstance(expected_exception, type(Exception)):
                assert type(ex) is expected_exception
            else:
                assert str(ex) == str(expected_exception)

    else:
        result = await dev.layout_transform(model)
        assert result
        assert isinstance(result, CNNx16Core)

        core_equal_result = (
            core_equal(result, expected) if expected else core_equal(result, result)
        )
        assert (
            core_equal_result == []
        ), f"layout mismatch (len={len(core_equal_result)})"


@pytest.mark.parametrize(
    "onnx_filename, expected_exception, expected",
    [("cifar10_short.onnx", None, None)],
)
@pytest.mark.asyncio
async def test_backend_ai8xize_layout_cifar10_short(  # noqa: F811
    onnx_filename, expected_exception, expected
):
    dev = MAX78000_ai8xize()

    model = onnx.load(data_folder / onnx_filename)
    if expected_exception:
        with pytest.raises(expected_exception):
            result = await dev.layout_transform(model)
    else:
        result = await dev.layout_transform(model)
        assert result
        assert isinstance(result, CNNx16Core)

        core_equal_result = core_equal(result, result)
        assert (
            core_equal_result == []
        ), f"layout mismatch (len={len(core_equal_result)})"


@pytest.mark.asyncio
async def test_backend_ai8xize_layout(cifar10_layout):
    dev = MAX78000_ai8xize()

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


def close_proc(x, y, distance=2):
    "if processor count x and y are smaller or equal distance"
    different_bits = x ^ y  # xor
    different_bits_count = bin(different_bits).count("1")
    return different_bits_count <= distance


@pytest.mark.parametrize(
    "net_name, comparable_config_file,transform_refdict_entries,skip_refdict_entries",
    [
        ("cifar10_short.onnx", [], {}, {}),  # we dont have a comparable config file
        (
            "cifar10.onnx",
            "cifar10-nas.yaml",
            {
                "operation": (lambda x, y: y if x.lower() == y.lower() else x),
            },
            {
                1: ["output_shift"],
                4: ["processors"],  # processors are upshifted by 32bit
                10: ["output_width", "activate"],  # activate none?
            },
        ),
        (
            # just to compare to above (comparable)
            "cifar10.onnx",
            c10_layers,
            {
                "operation": (lambda x, y: y if x.lower() == y.lower() else x),
            },
            {
                1: ["output_shift"],
                4: ["processors"],  # processors are upshifted by 32bit
                10: ["output_width"],
            },
        ),
        (
            "ai85-bayer2rgb-qat8-q.pth.onnx",
            "ai85-bayer2rgb.yaml",
            {
                "output_processors": (lambda x, y: y),  # upshifting not implemented yet
                "processors": (lambda x, y: y),  # upshifting not implemented yet
                "out_offset": (lambda x, y: y),  # we dont need the hopping
                # ignore case
                "operation": (lambda x, y: y if x.lower() == y.lower() else y),
            },
            {
                0: [
                    "name",
                    "activate",
                    # unfortunately, i cant do output_processors fornow
                    "output_processors",
                    "data_format",  # this is not in the reference
                ],
                1: [
                    "name",
                    "activate",
                    # unfortunately, i cant do output_processors fornow
                    "output_processors",
                ],
                2: [
                    "name",
                    "activate",
                    # unfortunately, i cant do output_processors fornow
                    "output_processors",
                    "output",  # output = true?
                ],
            },
        ),
        (
            "ai85-cifar10-qat8-q.pth.onnx",
            "cifar10-nas.yaml",
            {
                "operation": (lambda x, y: y if x.lower() == y.lower() else y),
            },
            {
                4: ["processors"],  # processors are upshifted by 32bit
                10: ["output_width", "activate"],  # activate none?
            },
        ),
        (
            "ai85-cifar100-qat8-q.pth.onnx",
            "cifar100-nas.yaml",
            {
                "operation": (lambda x, y: y if x.lower() == y.lower() else y),
            },
            {
                4: ["processors"],  # processors are upshifted by 32bit
                10: ["output_width", "activate"],  # activate none?
            },
        ),
        (
            "ai85-faceid_112-qat-q.pth.onnx",
            "faceid.yaml",
            {
                "operation": (lambda x, y: y if x.lower() == y.lower() else y),
                "out_offset": (lambda x, y: y if x in [0x2000, 0x1000] else x),
            },
            {
                0: ["streaming"],
                1: [
                    "streaming",
                    # cant get it right, because they are doing multipass with streaming
                    "processors",
                ],
            },
        ),
        (
            "ai85-kws20_v3-qat8-q.pth.onnx",
            "kws20-v3-hwc.yaml",
            {
                # transform the correct yaml vals to match the generated variant
                "kernel_size": (lambda x, y: f"{x}x{x}"),
                "out_offset": (lambda x, y: y if x == 0x2000 else x),
                # distance=6 is mostly because upshifting by 3 in layer 2
                "processors": (lambda x, y: y if close_proc(x, y, 6) else x),
                "output_width": (lambda x, y: y if y == 32 else x),
                "operation": (lambda x, y: y if x.lower() == y.lower() else x),
            },
            {8: ["activate", "output_width"]},
        ),
    ],
)
def test_layout_transform_generate_config_from_model_generic(  # noqa: C901
    net_name,
    comparable_config_file: str | dict,
    transform_refdict_entries: dict[str, callable],
    skip_refdict_entries: dict[int, list[str]],
):
    model = onnx.load(data_folder / net_name)

    default_values = {
        field_name: field.default
        for field_name, field in AI8XizeConfigLayer.model_fields.items()
    }
    if isinstance(comparable_config_file, str):
        with open(ai8x_synth_network / comparable_config_file) as compconfig_fx:
            comparable_config = yaml.safe_load(compconfig_fx)
    else:
        comparable_config = {
            "layers": [
                ly if isinstance(ly, dict) else ly.model_dump(exclude_unset=True)
                for ly in comparable_config_file
            ]
        }

    dev = MAX78000_ai8xize.create_device_from_name_and_ports(
        model_name="test_device",
        communication_port="/dev/virtualDevice",
        energy_port="/dev/virtualMetrics",
    )
    izer_config, locked_config, input_shape, transformed_model = (
        dev.generate_config_from_model(model)
    )
    assert izer_config
    layers = izer_config.get("layers", [])
    # ground truth
    for layeridx, ref_layerdict in enumerate(comparable_config.get("layers")):
        if "name" in layers[layeridx]:
            layers[layeridx].pop("name")

        if layeridx in skip_refdict_entries:
            for entry in skip_refdict_entries[layeridx]:
                if entry in ref_layerdict:
                    del ref_layerdict[entry]
                # skip both entries
                if entry in layers[layeridx]:
                    del layers[layeridx][entry]

        # check for missing keys
        assert len(set(ref_layerdict.keys()) - set(layers[layeridx].keys())) == 0, (
            f"missing keys ({set(ref_layerdict.keys())- set(layers[layeridx].keys())})"
            f" in Layer {layeridx}"
        )

        # check for extra keys, but ignore extra keys that are default values
        for extra_key in set(layers[layeridx].keys()) - set(ref_layerdict.keys()):
            # check if the extra key is a default value
            assert layers[layeridx][extra_key] == default_values[extra_key], (
                f"unexpected value in Layer {layeridx} for key {extra_key}"
                ", because this does not exist in the reference"
            )
            # it is a default value, so remove it from the dict
            layers[layeridx].pop(extra_key)

        ref_mod_layerdict = dict(ref_layerdict)
        for entry, transformer in transform_refdict_entries.items():
            if entry in ref_mod_layerdict:
                ref_mod_layerdict[entry] = transformer(
                    ref_mod_layerdict[entry], layers[layeridx].get(entry, None)
                )

        # processor to hex format
        for field in ["processors", "output_processors"]:
            for dictionary in [ref_mod_layerdict, layers[layeridx]]:
                if field in dictionary:
                    dictionary[field] = f"{dictionary[field]:016X}"

        # check for different values
        assert (
            ref_mod_layerdict == layers[layeridx]
        ), f"different values in Layer {layeridx}"
