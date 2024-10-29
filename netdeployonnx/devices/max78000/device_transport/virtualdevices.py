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
import abc
import struct

from google.protobuf.internal.encoder import _VarintBytes

from netdeployonnx.devices.max78000.device_transport.protobuffers import main_pb2


class VirtualAIODevice(abc.ABC):
    @abc.abstractmethod
    async def read(self, count: int, *args, **kwargs) -> bytes:
        pass

    @abc.abstractmethod
    def write(self, data, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def drain(self):
        pass

    async def read_async(self, size):
        return await self.read(size)

    async def write_async(self, data):
        self.write(data)
        return await self.drain()

    def close(self):
        pass

    @property
    def in_waiting(self):
        return 0


class MeasureDevice(VirtualAIODevice):
    "https://github.com/analogdevicesinc/max78000-powermonitor/blob/main/main.c#L110"

    mode: str = ""  #  we need global state to keep track of the mode

    def __init__(self, measurement: dict[str, list[float]] = {}):
        self.idle_power = measurement.get("idle_power", [0.03] * 3)
        # kernel, input, input+inference
        self.active_power = measurement.get("active_power", [70.3, 69.5, 327.8])
        self.time = measurement.get("time", [20.8e-3, 268.3e-6, 1.6e-3])
        self.power = measurement.get("power", [0.1, 0.2, 0.3, 0.4])
        self.voltages = measurement.get("voltages", [3.3, 3.3, 3.3, 1.8])

    def write(self, data, *args, **kwargs):
        MeasureDevice.mode = data

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


class FullDevice(VirtualAIODevice):
    def __init__(self, crc_func: callable = lambda x: 0, *args, **kwargs):
        self.crc_func = crc_func
        self.data = []
        self.collected_data = b""

    async def read(self, count: int, *args, **kwargs) -> bytes:
        def emit_keepalive():
            msg = main_pb2.ProtocolMessage()
            msg.version = 2
            msg.keepalive.next_tick = 23
            return msg

        self.data.append(emit_keepalive())
        bindata = b""
        for i_d in range(len(self.data)):
            msg = self.data.pop(0)
            d = msg.SerializeToString()
            # replay the encoding of the embedded device
            run_length_encoding = _VarintBytes(len(d))
            crcbytes = bytes([self.crc_func(run_length_encoding + d)])
            bindata += run_length_encoding + d + crcbytes
        assert len(self.data) == 0

        return bindata

    async def drain(self):
        self.work_on_data()

    def handle_msg(self, msg: main_pb2.ProtocolMessage): ...

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
            self.handle_msg(msg)
            ans_msg = main_pb2.ProtocolMessage(
                version=2,
                ack=main_pb2.ACK(),
                sequence=msg.sequence,
            )
            assert ans_msg.WhichOneof("message_type") == "ack"
            self.data.append(ans_msg)
            # remove this from the input
            self.collected_data = self.collected_data[len(packet) + 2 :]

    def write(self, data, *args, **kwargs):
        # virtual write means receive on device
        self.collected_data += data
        self.work_on_data()
