import asyncio
import struct
from pathlib import Path
from unittest import mock

import onnx
import pytest
from google.protobuf.internal.encoder import _VarintBytes

from netdeployonnx.devices.max78000 import MAX78000Metrics
from netdeployonnx.devices.max78000.ai8xize import (
    MAX78000_ai8xize,
)
from netdeployonnx.devices.max78000.device_transport.protobuffers import (
    main_pb2,
)


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


class MeasureDevice:
    "https://github.com/analogdevicesinc/max78000-powermonitor/blob/main/main.c#L110"

    def __init__(self, measurement: dict[str, list[float]] = {}):
        self.mode = ""
        self.idle_power = measurement.get("idle_power", [0.03] * 3)
        # kernel, input, input+inference
        self.active_power = measurement.get("active_power", [70.3, 69.5, 327.8])
        self.time = measurement.get("time", [20.8e-3, 268.3e-6, 1.6e-3])
        self.power = measurement.get("power", [0.1, 0.2, 0.3, 0.4])
        self.voltages = measurement.get("voltages", [3.3, 3.3, 3.3, 1.8])

    def write(self, data, *args, **kwargs):
        self.mode = data

    async def read(self, count: int, *args, **kwargs) -> bytes:
        # 3.3V, CA, CB, 1.8V
        COREA_IDX = 1  # noqa: N806, F841
        idle_power = self.idle_power
        active_power = self.active_power
        time = self.time
        power = self.power
        voltages = self.voltages
        if self.mode == b"v":
            # voltage mode
            sepstr = ",".join([f"{voltage:g}" for voltage in voltages])
            return bytes(f"{sepstr}\r\n", "utf8")
        elif self.mode in [b"t", b"c"]:
            # trigger mode CNN
            seps = []
            for i in range(3):
                vals = [
                    (active_power[i] - idle_power[i]) * time[i] / 1000.0,
                    time[i],
                    idle_power[i] / 1000.0,
                    active_power[i] / 1000.0,
                ]
                seps.append(",".join([f"{val:g}" for val in vals]))
            sepstr = ",".join(seps)
            return bytes(f"{sepstr}\r\n", "utf8")
        elif self.mode == b"s":
            # trigger mode System
            seps = []
            for i in range(3):
                vals = [
                    (idle_power[i]) * time[i] / 1000.0,
                    time[i],
                    idle_power[i] / 1000.0,
                ]
                seps.append(",".join([f"{val:g}" for val in vals]))
            sepstr = ",".join(seps)
            return bytes(f"{sepstr}\r\n", "utf8")
        elif self.mode == b"\x16":  # CTRL-V
            PM_VERSION = "EMULATED"  # noqa: N806
            __TIMESTAMP__ = "2024-08-21"  # noqa: N806
            AI_DEVICE = "EMULATED"  # noqa: N806
            return bytes(
                f"\r\nPMON {PM_VERSION} {__TIMESTAMP__} FOR {AI_DEVICE}\r\n\r\n", "utf8"
            )
        elif self.mode == b"i":
            # current mode
            sepstr = ",".join(
                [
                    f"{power[i] / voltages[i]:g}" if voltages[i] else f"{0.0:g}"
                    for i, voltage in enumerate(voltages)
                ]
            )
            return bytes(f"{sepstr}\r\n", "utf-8")
        elif self.mode == b"w":
            # power mode
            sepstr = ",".join(
                [f"{power[i]/1000.0:g}" for i, voltage in enumerate(voltages)]
            )
            return bytes(f"{sepstr}\r\n", "utf-8")
        return b""

    async def drain(
        self,
    ):
        pass


class FullDevice:
    def __init__(self, *args, **kwargs):
        self.data = []
        self.collected_data = b""

    async def read(self, count: int, *args, **kwargs) -> bytes:
        async def emit_keepalive():
            msg = main_pb2.ProtocolMessage()
            msg.version = 2
            msg.keepalive.next_tick = 23
            await asyncio.sleep(0.001)
            return msg.SerializeToString()

        self.data.append(await emit_keepalive())
        bindata = b""
        for i_d in range(len(self.data)):
            d = self.data.pop(0)
            run_length_encoding = _VarintBytes(len(d))
            bindata += run_length_encoding + d
        assert len(self.data) == 0

        return bindata

    async def drain(self):
        self.work_on_data()

    def work_on_data(self):
        "basically send ACKs"
        additional_bytes = 0
        packet = b""
        if len(self.collected_data) > 2:
            (additional_bits,) = struct.unpack("<H", self.collected_data[:2])
            additional_bytes = 2
            readlen = additional_bits // 8
            packet = self.collected_data[additional_bytes : additional_bytes + readlen]
        if len(packet) > 0:
            msg = main_pb2.ProtocolMessage.FromString(packet)
            ans_msg = main_pb2.ProtocolMessage(
                version=2,
                ack=main_pb2.ACK(),
                sequence=msg.sequence,
            )
            assert ans_msg.WhichOneof("message_type") == "ack"
            self.data.append(ans_msg.SerializeToString())
            # remove this from the input
            self.collected_data = self.collected_data[len(packet) + 2 :]

    def write(self, data, *args, **kwargs):
        # virtual write means receive on device
        self.collected_data += data
        self.work_on_data()


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
async def test_backend_ai8xize_execute_cifar10(
    open_serial_connection_virtual_device,
):
    with mock.patch(
        "serial_asyncio.open_serial_connection",
        open_serial_connection_virtual_device,
    ) as mock_open_serial_connection:  # noqa: F841
        dev = MAX78000_ai8xize()

        data_folder = Path(__file__).parent / "data"
        model = onnx.load(data_folder / "cifar10.onnx")
        layout = await dev.layout_transform(model)
        instr = await dev.compile_instructions(layout)
        res = await dev.execute(
            instructions=instr, metrics=MAX78000Metrics("/dev/null")
        )

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
async def test_backend_ai8xize_real_execute_exampledata(
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
        assert res
