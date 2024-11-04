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
import time

from aioserial import AioSerial

from netdeployonnx.devices.max78000.device_transport.protobuffers import main_pb2


async def read_serial(port, baudrate, buffer_size=1024):
    ser = AioSerial(port=port, baudrate=baudrate)

    while True:
        start = time.monotonic()
        data = await ser.read_async(ser.in_waiting)
        stop = time.monotonic()
        print(f"reader {stop-start:.2f}")
        if data:
            print(f"Received: {data}")

        msg = main_pb2.ProtocolMessage(version=2, sequence=1)
        msg.action.execute_measurement = main_pb2.ActionEnum.ASSERT_WEIGHTS
        await ser.write_async(msg.SerializeToString())

        start = time.monotonic()
        data = await ser.read_async(ser.in_waiting)
        stop = time.monotonic()
        print(f"reader ACK {stop-start:.2f}")


async def main():
    await read_serial("/dev/ttyUSB0", 1_500_000, buffer_size=8)


if __name__ == "__main__":
    asyncio.run(main())
