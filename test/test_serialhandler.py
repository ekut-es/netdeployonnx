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
from netdeployonnx.devices.max78000.device_transport.serialhandler import (
    DataHandler,
    crc,
    generate_table,
    recalc_crc,
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
        {
            "stage": "cnn_load_weights",
            "instructions": [
                ("ACTION", 40, 0),
                (
                    0,
                    b"0" * 40000,
                ),
            ],
        },
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
        # communication_port="/dev/ttyUSB0", energy_port="/dev/ttyACM0"
        communication_port="/dev/virtualDevice",
        energy_port="/dev/virtualEnergy",
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


def to_bytes(string: str) -> bytes:
    """converts 0A 03    to b'\x0a\x03'"""
    return bytes.fromhex(string.replace(" ", ""))


@pytest.mark.parametrize(
    "datastream, result",
    [
        (to_bytes("00 64"), []),
        (
            to_bytes(
                """04 C0 55 FF FF 12 08 02 22 0E 12 0C 08 80 80 83 82 05 12 04 16 B0 FF FF 08 02 22 0C 12 0A 08 80 80 12 02 73 6B 08 02 12 07 08 C0 56 10 64 28 34 08 02 12 05 08 C1 56 10 64 08 02 12 05 08 C2 56 10 64 08 02 12 05 08 C3 56 10 64 08 02 12 05 08 C4 56 10 64 08 02 12 05 08 C5 56 10 64 08 02 12 05 08 C6 56 10 64"""  # noqa E501
            ),
            [
                main_pb2.ProtocolMessage(
                    version=2,
                    payload=main_pb2.Payload(memory=[main_pb2.SetMemoryContent()]),
                ),
                main_pb2.ProtocolMessage(
                    version=2,
                    keepalive=main_pb2.Keepalive(
                        ticks=11078, next_tick=100, outqueue_size=52
                    ),
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
                main_pb2.ProtocolMessage(
                    version=2, keepalive=main_pb2.Keepalive(ticks=11078, next_tick=100)
                ),
            ],
        ),
        (
            to_bytes(
                """08 02 22 0E 12 0C 08 80 80 81 82 05 12 04 C0 55 FF FF 08 02 22 0E 12 0C 08 80 80 83 82 05 12 04 16 B0 FF FF 08 02 22 0C 12 0A 08 80 80 85 82 05 12 02 73 6B 08 02 12 07 08 B4 6F 10 64 28 34"""  # noqa E501
            ),
            [
                main_pb2.ProtocolMessage(
                    version=2, configuration=main_pb2.Configuration()
                ),
                main_pb2.ProtocolMessage(
                    version=2, configuration=main_pb2.Configuration()
                ),
                main_pb2.ProtocolMessage(
                    version=2, configuration=main_pb2.Configuration()
                ),
                main_pb2.ProtocolMessage(
                    version=2,
                    keepalive=main_pb2.Keepalive(
                        ticks=14260, next_tick=100, outqueue_size=52
                    ),
                ),
            ],
        ),
    ],
)
def test_search_protobuf_messages(datastream, result):
    d = DataHandler(None, None)
    msgs, datastream = d.search_protobuf_messages(datastream)
    assert len(msgs) == len(result), str(msgs)
    assert msgs == result, "nope"


@pytest.mark.parametrize(
    "msg",
    [
        main_pb2.ProtocolMessage(
            version=2,
            action=main_pb2.Action(
                execute_measurement=main_pb2.ActionEnum.MEASUREMENT,
                action_argument=4 << 1 + 0,  # this is PCLK + CLKDIV 1
            ),
        ),
        main_pb2.ProtocolMessage(
            version=2,
            action=main_pb2.Action(execute_measurement=main_pb2.ActionEnum.NONE),
        ),
        main_pb2.ProtocolMessage(
            version=2,
            keepalive=main_pb2.Keepalive(ticks=12345),
        ),
        main_pb2.ProtocolMessage(
            version=2,
            payload=main_pb2.Payload(
                memory=[
                    main_pb2.SetMemoryContent(
                        address=12345,
                        data=b"asdf" * 4,
                        setAddr=True,
                    )
                ],
            ),
        ),
        main_pb2.ProtocolMessage(
            version=2,
            payload=main_pb2.Payload(
                memory=[
                    main_pb2.SetMemoryContent(
                        address=12345,
                        data=b"asdf" * 200,  # 800 bytes
                        setAddr=True,
                    )
                ],
            ),
        ),
    ],
)
def test_crc(msg):
    # import debugpy; debugpy.listen(4567);debugpy.wait_for_client();debugpy.breakpoint() # noqa E501

    msg.checksum = recalc_crc(msg)

    serdata = msg.SerializeToString()
    print(">", ",".join([f"0x{b:02X}" for b in serdata]), len(serdata))
    new_crc = crc(serdata)
    print(f"CRC={new_crc:08X}")
    # assert new_crc == 0x0000_0000 # 0xFFFF_FFFF


@pytest.mark.parametrize(
    "data",
    [
        b"\t\x08\x02\x12\x05\x08\x83\r\x10\x05\xee",
    ],
)
def test_find_with_rle_and_crc8(data):
    dh = DataHandler(None, None)
    msgs, datastream = dh.search_protobuf_messages(data)
    assert datastream == b""
    assert len(msgs) == 1
    assert msgs[0].version == 2
    assert msgs[0].keepalive.ticks >= 0

    print(msgs[0])


@pytest.mark.skip("only generate once")
def test_generate_table():
    print(generate_table(lambda i: crc(bytes([i]))))


@pytest.mark.parametrize(
    "data, expect_error",
    [
        # this was without crc
        (bytes([0x9, 0x8, 0x3, 0x22, 0x5, 0x8, 0xDE, 0x74, 0x10, 0x5]), True),
        # this is not really parsable
        (
            bytes(
                [
                    0x5,
                    0x98,
                    0x8,
                    0x8,
                    0x3,
                    0x22,
                    0x4,
                    0x8,
                    0x4,
                    0x10,
                    0x5,
                    0x8E,
                    0x8,
                    0x8,
                    0x3,
                    0x22,
                    0x4,
                    0x8,
                    0x5,
                    0x10,
                ]
            ),
            True,
        ),
        # this should be good, but it isnt because the second message is faulty
        (
            bytes(
                [
                    0x8,
                    0x8,
                    0x3,
                    0x22,
                    0x4,
                    0x8,
                    0x4,
                    0x10,
                    0x5,
                    0x8E,
                    0x8,
                    0x8,
                    0x3,
                    0x22,
                    0x4,
                    0x8,
                    0x5,
                    0x10,
                ]
            ),
            True,
        ),
        # this should be good
        (bytes([0x8, 0x8, 0x3, 0x22, 0x4, 0x8, 0x4, 0x10, 0x5, 0x8E]), False),
    ],
)
def test_serial_messages(data, expect_error: bool):
    dh = DataHandler(None, None)
    msgs, datastream = dh.search_protobuf_messages(data)
    try:
        assert datastream == b""
        assert len(msgs) == 1
        assert msgs[0].version == 3
        assert msgs[0].keepalive.ticks >= 0
    except AssertionError as ae:
        if expect_error:
            return True
        else:
            raise ae

    print(msgs[0])
