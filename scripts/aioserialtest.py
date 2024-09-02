import asyncio
import time

from aioserial import AioSerial


async def read_serial(port, baudrate, buffer_size=1024):
    ser = AioSerial(port=port, baudrate=baudrate)

    while True:
        start = time.monotonic()
        data = await ser.read_async(buffer_size)
        stop = time.monotonic()
        print(f"reader {stop-start:.2f}")
        if data:
            print(f"Received: {data}")


async def main():
    await read_serial("/dev/ttyUSB0", 1_500_000, buffer_size=8)


if __name__ == "__main__":
    asyncio.run(main())
