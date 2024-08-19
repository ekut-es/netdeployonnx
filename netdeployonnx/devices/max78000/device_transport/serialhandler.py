import asyncio
import logging
import statistics
import struct
import time
import traceback
from collections.abc import Awaitable
from typing import Callable

import google.protobuf.message
import pytest
from crc import Calculator, Crc32  # pip install crc
from fastcrc import crc8  # pip install fastcrc
from google.protobuf.internal.decoder import _DecodeVarint  # pip install protobuf
from serial_asyncio import (
    open_serial_connection,  # pip install pyserial-asyncio
)

from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.protobuffers import (
    main_pb2,
)

BUFFER_READ_SIZE = 1024
BUFFER_COMPARE_SIZE = 1000
DEFAULT_TIMEOUT = 1.1

MessageHandler = Callable[
    [main_pb2.ProtocolMessage, asyncio.StreamWriter], Awaitable[bool]
]  # message, writer -> bool

crc_calc = Calculator(Crc32.POSIX, optimized=True)

POLY = 0x04C11DB7  # POSIX / bzip2 /jamcrc / mpeg_2


def crc8(data):  # noqa F811 (redefinition of crc8)
    polynomial = 0x07
    crc = 0

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:  # noqa SIM108
                crc = (crc << 1) ^ polynomial
            else:
                crc = crc << 1

    return crc & 0xFF


def crc(data: bytes) -> int:
    # return crc_calc.checksum(data)
    # return crc32.mpeg_2(data)
    return crc8(data)


def recalc_crc(msg: "main_pb2.ProtocolMessage") -> int:
    for crafted in range(256):
        msg.checksum = crafted
        finalmsg = msg.SerializeToString()
        # print(crc_appended)
        new_crc = crc(finalmsg)
        if new_crc == 0:
            break
    # print(f"{crafted} => {new_crc:08X}", finalmsg)
    return crafted
    # return struct.unpack("<I", struct.pack(">I", crafted))[0]


class KeepaliveTimer:
    def __init__(self):
        self.timer = 0
        self.timer_max_val = 10  # default 10 ticks @ 100ms => 1s
        self.initialized = False
        self.warning = False
        self.last_reset = time.time()
        self.keepalive_list = []
        self.keepalive_inqueue = []
        self.keepalive_outqueue = []
        self.task_timer = None
        self.task_stats = None

    def init_once(self, timeout_in_s: float):
        if not self.initialized:
            self._init(timeout_in_s * 10.0)
            self.initialized = True

    def _init(self, timer_max_val: int):
        self.timer_max_val = timer_max_val
        self.task_timer = asyncio.create_task(self.timer_task())
        self.task_stats = asyncio.create_task(self.print_statistics())

    async def timer_task(self):
        while self.timer < self.timer_max_val:
            self.timer += 1
            await asyncio.sleep(0.1)
            if not self.warning and (self.timer / self.timer_max_val) > 0.5:
                logging.warning("Keepalive hickup ~ 50%")
                self.warning = True
        # raise asyncio.TimeoutError("Keepalive-Timer ran out")

    async def print_statistics(self):
        try:
            while self.task_timer is not None:
                await asyncio.sleep(1)
                if len(self.keepalive_list) > 0:
                    if len(self.keepalive_inqueue) == 0:
                        self.keepalive_inqueue = [0]
                    if len(self.keepalive_outqueue) == 0:
                        self.keepalive_outqueue = [0]
                    v_min, v_mean, v_max = (  # noqa F841
                        min(self.keepalive_list),
                        statistics.mean(self.keepalive_list),
                        max(self.keepalive_list),
                    )
                    in_min, in_mean, in_max = (  # noqa F841
                        min(self.keepalive_inqueue),
                        statistics.mean(self.keepalive_inqueue),
                        max(self.keepalive_inqueue),
                    )
                    out_min, out_mean, out_max = (  # noqa F841
                        min(self.keepalive_outqueue),
                        statistics.mean(self.keepalive_outqueue),
                        max(self.keepalive_outqueue),
                    )
                    self.keepalive_list = []
                    self.keepalive_inqueue = []
                    self.keepalive_outqueue = []
                    print(
                        f"Keepalive: (min={v_min:2.2f}, mean={v_max:2.2f}, "
                        f"max={v_max:2.2f}"
                        f"[I={in_mean:2.2f}/O={out_mean:2.2f}])"
                    )
        except Exception:
            traceback.print_exc()

    def check_and_raise(self):
        if self.timer >= self.timer_max_val:
            self.task_timer = None
            raise asyncio.TimeoutError("Keepalive-Timer ran out")

    def reset(self):
        now = time.time()
        if (now - self.last_reset) > 0.005:
            self.keepalive_list.append(round((now - self.last_reset) * 1000, 2))
        self.last_reset = time.time()
        self.timer = 0
        self.warning = False


warning_flags = {
    "NONE": 0,
    "QUEUE_WARNING": 0x01,
    "DECODE_WARNING": 0x02,
    "IOREAD_WARNING": 0x04,
    "CHECKSUM_WARNING": 0x08,
    "QUEUE_SKIP_WARNING": 0x10,
    "TX_RETRIGGER": 0x1000,
    "RX_RETRIGGER": 0x2000,
    "SET_MEMORY_WARNING_DIVIDE_BY_4": 0x4000,
}

inverted_warning_flags = {value: key for key, value in warning_flags.items()}


class DataHandler:
    def __init__(self, reader, writer, debug: bool = False):
        self.reader = reader
        self.writer = writer
        self.debug = debug
        self.handlers: list[MessageHandler] = [self.keepalive_handler]

        self.msgs = []
        self.external_send_queue = []
        self.datastream = b""
        self.keepalive_answer = main_pb2.ProtocolMessage()
        self.keepalive_answer.keepalive.ticks = 0
        self.keepalive_timer = KeepaliveTimer()

        self.last_tick_id = 0

        self.open_futures = []

        debug_data = []
        for i in range(255 * 16):
            debug_data.extend([(i & 0xF00) >> 8, i & 0xFF])
        self.debug_data = bytes(debug_data)

    async def keepalive_handler(self, msg, writer) -> bool:
        if msg.WhichOneof("message_type") == "keepalive":
            try:
                for warning_id, warning_text in inverted_warning_flags.items():
                    if msg.keepalive.warning & warning_id:
                        logging.warning(
                            f"[{msg.keepalive.ticks:06d}] Device: {warning_text}"
                        )
                future = self.open_futures.pop(0) if len(self.open_futures) else None
                if future:
                    # check if this is really the next keepalive
                    if msg.keepalive.ticks >= (future.last_tick + 1):
                        # we dont have a warning: 0, else warning value
                        future.set_result(msg.keepalive.warning & 0x0F)
                    else:
                        ...
                        # print("interjection?!?")
                        # print(future.last_tick, self.last_tick_id,
                        #       msg.keepalive.ticks)
                        # # assume it was false
                        future.set_result(0x4000)

                logging.debug(
                    f"[ALIVE n={msg.keepalive.next_tick}] {msg.keepalive.ticks}"
                    f" [I: {msg.keepalive.inqueue_size:03d} "
                    f"/ O: {msg.keepalive.outqueue_size:03d}]"
                )
                if msg.keepalive.ticks > 0:
                    self.last_tick_id = msg.keepalive.ticks
                # maybe answer?
                self.keepalive_answer.keepalive.ticks += 1
                self.keepalive_timer.reset()
                self.keepalive_timer.keepalive_inqueue.append(
                    msg.keepalive.inqueue_size
                )
                self.keepalive_timer.keepalive_outqueue.append(
                    msg.keepalive.outqueue_size
                )
                # await self.send_msg(self.keepalive_answer)
            except Exception:
                traceback.print_exc()
            finally:
                return True
        return False

    async def default_handle_msg(self, msg, writer) -> bool:
        print(f"[? {msg.WhichOneof('message_type')} ]{str(msg)[:600]}")

    async def send_msg(self, msg: "main_pb2.ProtocolMessage"):
        sent = asyncio.Future()
        self.external_send_queue.append((msg, sent))
        return await sent

    async def handle_sendqueue(self, writer) -> None:
        # sending
        if self.debug:
            ret1 = writer.write(self.debug_data)
            ret2 = await writer.drain()
        elif len(self.external_send_queue) > 0:
            msg, future = self.external_send_queue.pop(0)
            if msg:
                if msg is None:
                    future.set_result(16)
                    return
                # if msg.WhichOneof('message_type') == 'payload':
                #     if len(msg.payload.registers):
                #         print(len(msg.payload.registers))
                #     if len(msg.payload.memory):
                #         print(sum(len(mem.data) for mem in msg.payload.memory),msg.payload) # noqa E501

                # carefully craft the crc of the message to be 0x0
                msg.checksum = recalc_crc(msg)
                serdata = msg.SerializeToString()
                if not msg.keepalive.ticks:
                    logging.debug(f"sent msg [len={len(serdata)}]")
                if len(serdata) > BUFFER_COMPARE_SIZE:
                    future.set_result(32)
                    # we dont want to overload the smol buffer
                    # (cant increase it because DMA)
                    return
                # run length encoding
                future.last_tick = self.last_tick_id
                ret0 = writer.write(  # noqa F841
                    struct.pack("<H", len(serdata) * 8)
                )  # in bits, to check if we received garbage
                ret1 = writer.write(serdata)  # noqa F841
                ret2 = await writer.drain()  # noqa F841
                self.open_futures.append(future)
        else:
            # nothing to do, but send a keepalive back
            ...

    async def next_msg(self, timeout=DEFAULT_TIMEOUT):
        if len(self.msgs) > 0:
            return self.msgs.pop(0)
        self.keepalive_timer.init_once(timeout)
        self.datastream += await asyncio.wait_for(
            self.reader.read(BUFFER_READ_SIZE), timeout
        )  # should return after reading 1 byte
        msgs, self.datastream = self.search_protobuf_messages(self.datastream)
        self.msgs += msgs
        if len(self.msgs) > 0:
            return self.msgs.pop(0)

    async def receive(self, message_filter=lambda msg: False, timeout=1):
        future_until_called = asyncio.Future()

        async def specific_message_handler(msg, writer) -> bool:
            if message_filter(msg):
                future_until_called.set_result(msg)
                return True
            return False

        try:
            self.handlers.insert(0, specific_message_handler)
            await asyncio.wait_for(future_until_called, timeout=timeout)
            return future_until_called.result()
        except asyncio.TimeoutError:
            ...  # shit happens, did not receive a message in time
        finally:
            self.handlers.remove(specific_message_handler)
        return None

    def ParseFromStream(self, data: bytes) -> main_pb2.ProtocolMessage:  # noqa N802
        ret = main_pb2.ProtocolMessage()
        try:
            # read delimited
            length, new_pos = _DecodeVarint(data, 0)
            ret.ParseFromString(data[new_pos : new_pos + length])
        except google.protobuf.message.DecodeError as ex:
            raise Exception() from ex
        return ret

    def search_protobuf_messages(
        self, datastream: bytes, discard_limit: int = 100
    ) -> tuple[list, bytes]:
        messages = []
        loop_while = True
        while loop_while:
            # Find the start of the next protobuf message
            # print(f"[{len(datastream)}] ", " ".join([f"{b:02X}" for b in datastream]))
            for startindex in range(len(datastream)):
                if (
                    len(datastream[startindex:]) < 4
                ):  # min size is 1 byte length, one version, one tick, one end
                    loop_while = False
                    break  # cant work
                try:
                    # Deserialize the message
                    message = self.ParseFromStream(datastream[startindex:])
                    if message.ByteSize() > 0:
                        messages.append(message)
                        if startindex > 1:
                            logging.warning(f"skipped data: {datastream[0:startindex]}")
                        # Update the datastream to point to the next message
                        datastream = datastream[startindex + message.ByteSize() :]
                        break
                except ValueError:
                    continue
                except Exception:
                    # If parsing fails, continue searching from the next position
                    # print(f"startindex {startindex} did not work", type(e), e)
                    ...
        return messages, datastream

    async def find_message_handler(self, msg, writer) -> bool:
        for handler in self.handlers:
            try:
                if await handler(msg, writer):
                    return True
            except Exception:
                ...
                # traceback.print_exc()
        return await self.default_handle_msg(msg, writer)


async def handle_serial(
    commands: Commands, tty: str, debug: bool = False, timeout: float = 1
):
    global data
    try:
        reader, writer = await open_serial_connection(url=tty, baudrate=1_500_000)
        data_handler = DataHandler(reader, writer, debug=debug)
        commands._register(data_handler)
        while not commands.set_exit_request:
            try:
                data_handler.keepalive_timer.check_and_raise()
                msg = await data_handler.next_msg(timeout=timeout)
                if msg:
                    try:
                        await data_handler.find_message_handler(msg, writer)
                        await data_handler.handle_sendqueue(writer)
                    except Exception:
                        traceback.print_exc()
            except TimeoutError as timeoutErr:
                if "Keepalive-Timer" in str(timeoutErr):
                    print("Timeout:", str(timeoutErr))
                    break
            except Exception:
                traceback.print_exc()
                break
    except Exception:
        traceback.print_exc()
    await asyncio.sleep(0.5)
    exit(0)


async def main(): ...


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
                execute_measurement=main_pb2.ActionEnum.MEASUREMENT_WITH_IPO
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


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
