import asyncio
import contextlib
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pytest

from netdeployonnx.devices.max78000 import MAX78000Metrics
from netdeployonnx.devices.max78000.ai8xize import (
    MAX78000_ai8xize,
)
from netdeployonnx.devices.max78000.device_transport.protobuffers import (
    main_pb2,
)
from netdeployonnx.devices.max78000.device_transport.virtualdevices import (
    FullDevice,
    MeasureDevice,
)

logging.basicConfig(
    level=logging.WARNING,
    format="[+{relativeCreated:2.2f}ms] {levelname}: ({funcName:10s}) {message}",
    style="{",
)


@pytest.fixture
def test_instructions():
    return [
        {
            "stage": "cnn_enable",
            "instructions": [("ACTION", main_pb2.ActionEnum.RUN_CNN_ENABLE, 0)],
        },
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


def print_chunks(origval, val_under_test, diff_vector):
    chunksize = 32

    def hexstr(x):
        return "".join(f"{h:02X}" for h in x)

    for chunk_idx in range(max(len(origval), len(val_under_test)) // chunksize):
        print(hexstr(origval[chunk_idx * chunksize : (chunk_idx + 1) * chunksize]))
        print(
            hexstr(val_under_test[chunk_idx * chunksize : (chunk_idx + 1) * chunksize])
        )
        print("=")
        print(hexstr(diff_vector[chunk_idx * chunksize : (chunk_idx + 1) * chunksize]))
        print()


@pytest.fixture(scope="module")
def open_serial_connection_virtual_device(
    full_devices: list[str] = ["/dev/ttyACM1", "/dev/ttyUSB0"],
    measure_devices: list[str] = ["/dev/ttyACM0", "/dev/null"],
):
    mdev = MeasureDevice()
    fdev = FullDevice()

    async def return_virtual_dev(url, *args, **kwargs):
        reader, writer = mock.AsyncMock(), mock.AsyncMock()
        if url in measure_devices:
            reader.read = mdev.read
            writer.drain = mdev.drain
            writer.write = mdev.write
        elif url in full_devices:
            reader.read = fdev.read
            writer.drain = fdev.drain
            writer.write = fdev.write
        else:
            raise ValueError("unknown device")
        writer.close = mock.MagicMock()
        reader.close = mock.MagicMock()
        return reader, writer

    return return_virtual_dev


@pytest.mark.asyncio
async def test_backend_ai8xize_run_onnx(open_serial_connection_virtual_device):
    with (
        mock.patch(
            "serial_asyncio.open_serial_connection",
            open_serial_connection_virtual_device,
        ),
        mock.patch(
            "netdeployonnx.devices.max78000.device_transport.serialhandler.open_serial_connection",
            open_serial_connection_virtual_device,
        ),  # noqa: F841
    ):  # noqa: F841
        dev = MAX78000_ai8xize(
            communication_port="/dev/ttyACM1", energy_port="/dev/ttyACM0"
        )
        assert dev

        data_folder = Path(__file__).parent / "data"
        model = onnx.load(data_folder / "cifar10.onnx")
        result = await dev.run_onnx(model, None)
        print(result)
        # assert mock_open_serial_connection.assert_awaited()
        assert "exception" not in result


@pytest.mark.asyncio
async def test_backend_ai8xize_execute_cifar10_riched(
    open_serial_connection_virtual_device,
):
    with mock.patch(
        "serial_asyncio.open_serial_connection",
        open_serial_connection_virtual_device,
    ) as mock_open_serial_connection:  # noqa: F841
        dev = MAX78000_ai8xize(
            communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
        )

        data_folder = Path(__file__).parent / "data"
        model = onnx.load(data_folder / "cifar10.onnx")
        layout = await dev.layout_transform(model)
        instr = await dev.compile_instructions(layout)
        res = await dev.execute(
            instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
        )

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res


@pytest.mark.asyncio
async def test_backend_ai8xize_execute_cifar10_unriched(
    open_serial_connection_virtual_device,
):
    progress_obj_mock = mock.MagicMock()

    @contextlib.contextmanager
    def progress_mock(*args):
        yield progress_obj_mock

    progress_obj_mock.add_task = lambda *args, **kwargs: (args, kwargs)
    progress_obj_mock.advance = print
    with (
        mock.patch("netdeployonnx.devices.max78000.device.Progress", progress_mock),
        mock.patch(
            "serial_asyncio.open_serial_connection",
            open_serial_connection_virtual_device,
        ) as mock_open_serial_connection,  # noqa: F841
    ):  # noqa: F841
        dev = MAX78000_ai8xize(
            communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
        )

        data_folder = Path(__file__).parent / "data"
        model = onnx.load(data_folder / "cifar10.onnx")
        layout = await dev.layout_transform(model)
        instr = await dev.compile_instructions(layout)
        res = await dev.execute(
            instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
        )

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res


@pytest.mark.asyncio
async def test_backend_ai8xize_execute_fakedata(
    open_serial_connection_virtual_device,
):
    async def send_batch_patched(
        self, msgs: list[main_pb2.ProtocolMessage]
    ) -> list[int]:
        """
        returns errorcode, 0 if success
        """
        if self.dataHandler:
            if msgs:
                return await self.dataHandler.send_msgs(msgs)
            else:
                return [0]
        else:
            return [1024]  # no data handler

    with (
        mock.patch(
            "serial_asyncio.open_serial_connection",
            open_serial_connection_virtual_device,
        ),
        mock.patch(
            "netdeployonnx.devices.max78000.device_transport.serialhandler.open_serial_connection",
            open_serial_connection_virtual_device,
        ),
    ):  # noqa: F841
        dev = MAX78000_ai8xize(
            communication_port="/dev/ttyACM1", energy_port="/dev/ttyACM0"
        )

        instr = [
            {
                "stage": "cnn_enable",
                "instructions": [
                    ("", "NONE", ""),
                    ("", "NONE", ""),
                    ("", "NONE", ""),
                    ("", "NONE", ""),
                ],
            },
        ]
        with mock.patch(
            "netdeployonnx.devices.max78000.device_transport.commands.Commands.send_batch",
            side_effect=lambda msgs: send_batch_patched(dev.commands, msgs),
        ) as mock_send_batch:
            res = await dev.execute(
                instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
            )

            dev.commands.exit_request()
            await asyncio.sleep(0.1)  # wait for exit
            assert res
            expected_args = [
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                ],
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                ],
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                ],
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                ],
            ]
            for i, (args, kwargs) in enumerate(mock_send_batch.await_args_list):
                assert expected_args[i] == list(args[0])


@pytest.mark.asyncio
async def test_backend_ai8xize_real_virtual_execute_exampledata_patched(
    test_instructions,
):
    dev = MAX78000_ai8xize(
        communication_port="/dev/virtualDevice", energy_port="/dev/virtualEnergy"
    )

    instr = test_instructions
    progress_obj_mock = mock.MagicMock()

    @contextlib.contextmanager
    def progress_mock(*args):
        yield progress_obj_mock

    progress_obj_mock.add_task = lambda *args, **kwargs: (args, kwargs)
    progress_obj_mock.advance = print
    with (
        mock.patch("netdeployonnx.devices.max78000.device.Progress", progress_mock),
    ):
        try:
            res = await dev.execute(
                instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
            )
        except TimeoutError:
            raise TimeoutError("timeout")

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res == []
        # TODO: check for [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
async def test_backend_ai8xize_real_execute_exampledata_unpatched(
    test_instructions,
):
    dev = MAX78000_ai8xize(
        communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
    )

    instr = test_instructions
    try:
        res = await dev.execute(
            instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
        )
    except TimeoutError:
        raise TimeoutError("timeout")

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res


@pytest.mark.asyncio
async def test_backend_ai8xize_real_execute_exampledata_patched(
    test_instructions,
):
    dev = MAX78000_ai8xize(
        communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
    )
    instr = test_instructions
    progress_obj_mock = mock.MagicMock()

    @contextlib.contextmanager
    def progress_mock(*args):
        yield progress_obj_mock

    progress_obj_mock.add_task = lambda *args, **kwargs: (args, kwargs)
    progress_obj_mock.advance = print
    with (
        mock.patch("netdeployonnx.devices.max78000.device.Progress", progress_mock),
    ):
        try:
            res = await dev.execute(
                instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
            )
        except TimeoutError:
            raise TimeoutError("timeout")

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res


@pytest.mark.asyncio
async def test_backend_ai8xize_virtual_execute_exampledata(
    test_instructions, open_serial_connection_virtual_device
):
    with (
        mock.patch(
            "serial_asyncio.open_serial_connection",
            open_serial_connection_virtual_device,
        ),
        mock.patch(
            "netdeployonnx.devices.max78000.device_transport.serialhandler.open_serial_connection",
            open_serial_connection_virtual_device,
        ),
    ):  # noqa: F841
        dev = MAX78000_ai8xize(
            communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
        )

        instr = test_instructions
        try:
            res = await dev.execute(
                instructions=instr, metrics=MAX78000Metrics(dev.energy_port)
            )
        except TimeoutError:
            raise TimeoutError("timeout")

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            await dev.handle_serial_task_closed

        await cleanup()
        assert res


@pytest.mark.asyncio
async def test_backend_ai8xize_virtual_runonnx_exampledata_patched(
    test_instructions,
):
    dev = MAX78000_ai8xize(
        communication_port="/dev/virtualDevice", energy_port="/dev/virtualEnergy"
    )

    progress_obj_mock = mock.MagicMock()

    @contextlib.contextmanager
    def progress_mock(*args):
        yield progress_obj_mock

    progress_obj_mock.add_task = lambda *args, **kwargs: (args, kwargs)
    progress_obj_mock.advance = print
    with (
        mock.patch("netdeployonnx.devices.max78000.device.Progress", progress_mock),
        mock.patch(
            "netdeployonnx.devices.max78000.ai8xize."
            "MAX78000_ai8xize.layout_transform"
        ) as mock_layout_transform,
        mock.patch(
            "netdeployonnx.devices.max78000.ai8xize."
            "MAX78000_ai8xize.compile_instructions"
        ) as mock_compile_instructions,
    ):
        mock_layout_transform.return_value = None
        mock_compile_instructions.return_value = test_instructions
        try:
            res = await dev.run_onnx(
                onnx.helper.make_model(
                    onnx.helper.make_graph(nodes=[], name="test", inputs=[], outputs=[])
                ),
                None,
            )
        except TimeoutError:
            raise TimeoutError("timeout")

        dev.commands.exit_request()
        await asyncio.sleep(0.1)  # wait for exit

        async def cleanup():
            if dev.handle_serial_task_closed:
                await dev.handle_serial_task_closed

        await cleanup()
        assert "exception" not in res
        assert res.pop("result") == []
        exec_times = res.pop("deployment_execution_times")
        assert exec_times
        assert all(
            np.isclose(exec_times[key], value, atol=0.1)
            for key, value in {
                "compile_instructions": 0.0,
                "execute": 0.01,
                "layout_transform": 0.01,
                "total": 0.01,
            }.items()
        ), "execution times are not as expected"
        assert res == {
            "uJ_per_all": 1986.05,
            "uJ_per_convolution": 505.79,
            "uJ_per_input_loading": 18.64,
            "uJ_per_weights_loading": 1461.62,
            "uW_per_all": 398040.0,
            "uW_per_convolution": 258300.0,
            "uW_per_input_loading": 69470.0,
            "uW_per_weights_loading": 70270.0,
            "us_per_all": 22400.0,
            "us_per_convolution": 1331.7,
            "us_per_input_loading": 268.3,
            "us_per_weights_loading": 20800.0,
        }
