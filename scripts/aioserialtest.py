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
