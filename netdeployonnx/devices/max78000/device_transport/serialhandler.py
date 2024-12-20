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
import statistics
import struct
import time
import traceback
from collections.abc import Awaitable
from typing import Callable

import aioserial
import google.protobuf.message
from crc import Calculator, Crc32  # pip install crc
from fastcrc import crc8  # pip install fastcrc
from google.protobuf.internal.decoder import _DecodeVarint  # pip install protobuf

from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.protobuffers import (
    main_pb2,
)
from netdeployonnx.devices.max78000.device_transport.virtualdevices import (
    FullDevice,
    MeasureDevice,
)

BUFFER_READ_SIZE = 1024
BUFFER_COMPARE_SIZE = 1000
DEFAULT_TIMEOUT = 1.1

MessageHandler = Callable[
    [main_pb2.ProtocolMessage, asyncio.StreamWriter], Awaitable[bool]
]  # message, writer -> bool

crc_calc = Calculator(Crc32.POSIX, optimized=True)
# console = rich.console.Console()

POLY = 0x04C11DB7  # POSIX / bzip2 /jamcrc / mpeg_2


def generate_table(crc_method) -> str:
    table = "static const uint8_t crc8_table[256] = {\n"
    for i in range(256):
        if i % 8 == 0:
            table += "    "
        table += f"0x{crc_method(i):02x}, "
        if i % 8 == 7:
            table += "\n"
    table += "};"
    return table


def crc8_own(data):
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
    # crcs = {
    #     name:getattr(crc8, name)(data)
    #     for name in crc8.__always_supported
    # }
    # return crcs
    return crc8.smbus(data)


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
        self.TIMER_TICK = 0.1
        self.timer = 0
        self.timer_max_val = 10  # default 10 ticks @ 100ms => 1s
        self.initialized = False
        self.warning = False
        self.last_reset = time.monotonic()
        self.keepalive_list = []
        self.keepalive_inqueue = []
        self.keepalive_outqueue = []
        self.task_timer = None
        self.task_stats = None

    async def close_tasks(self):
        if self.task_timer:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                self.task_timer.cancel()
                await self.task_timer
        if self.task_stats:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                self.task_stats.cancel()
                await self.task_stats

    def init_once(self, timeout_in_s: float):
        if not self.initialized:
            self._init(timeout_in_s / self.TIMER_TICK)
            self.initialized = True

    def _init(self, timer_max_val: int):
        self.timer_max_val = timer_max_val
        self.task_timer = asyncio.create_task(self.timer_task())
        self.task_stats = asyncio.create_task(self.print_statistics())

    async def timer_task(self):
        while self.timer < self.timer_max_val:
            elapsed = time.monotonic() - self.last_reset
            self.timer = int(elapsed / self.TIMER_TICK)  # TIMER_TICK is 0.1
            await asyncio.sleep(self.TIMER_TICK)
            # print(self.timer, self.timer_max_val,
            # (self.timer / self.timer_max_val)
            # )
            if not self.warning and (self.timer / self.timer_max_val) > 0.5:
                logging.warning("Keepalive hickup ~ 50%")
                self.warning = True

    async def print_statistics(self):
        try:
            while self.task_timer is not None:
                await asyncio.sleep(5)
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
                        f"Keepalive: (min={v_min:2.2f}, mean={v_mean:2.2f}, "
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
        now = time.monotonic()
        if (now - self.last_reset) > 0.005:
            self.keepalive_list.append(round((now - self.last_reset) * 1000, 2))
        self.last_reset = time.monotonic()
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
    "INTERRUPT_NMI": 0x010000,
    "INTERRUPT_HARDFAULT": 0x020000,
    "INTERRUPT_MEMMANAGE": 0x040000,
    "INTERRUPT_BUSFAULT": 0x080000,
    "INTERRUPT_USAGEFAULT": 0x100000,
    "INTERRUPT_WUT": 0x200000,
    "INTERRUPT_DEFAULTHANDLR": 0x400000,
    "INTERRUPT_CNN": 0x800000,
}

inverted_warning_flags = {value: key for key, value in warning_flags.items()}


class PacketOrderSender:
    MAX_QUEUE_SIZE = 1

    def __init__(self, data_handler: "DataHandler", one_message_timeout: float = 5.0):
        self.data_handler = data_handler
        self.expected_sequence = 1  # 0xFFFF_FFF0
        self.enqueued_sequence = self.expected_sequence
        self.one_message_timeout = one_message_timeout
        self.wait_queue = {}
        self.sequence_queue = {}
        self.sequence_success = {}
        self.sendqueue = []
        self.resend = False

    def enqueue(self, msg: main_pb2.ProtocolMessage) -> int:
        msg.sequence = self.enqueued_sequence
        msgstr = str(msg).replace("\n", " ")[:100]
        logging.info(f"enqueing msg with seq {msg.sequence} {msgstr}")
        self.sequence_queue[self.enqueued_sequence] = msg
        self.enqueued_sequence += 1
        return msg.sequence

    def accept_acknowledge(self, sequence: int, success: bool) -> bool:
        "returns true if the sequence is ok, else returns false and requests a resend"
        if self.expected_sequence == sequence:
            # accept it by removing it from the sendqueue
            if len(self.sendqueue) > 0:
                self.sendqueue.pop(0)
            else:
                raise Exception("when does this happen?")
            self.sequence_success[sequence] = success  # save the success
            if sequence in self.wait_queue:
                try:
                    self.wait_queue[sequence].set_result(success)
                except asyncio.exceptions.InvalidStateError as ise:
                    print(ise)
                    raise ise

            self.expected_sequence += 1
            return True
        else:
            self.resend = True
            return False

    async def work(self):
        "keep up a send queue and resend if necessary"
        if self.resend:
            # we need to resend the current sequence
            # 1. flush the send queue
            # 2. resend the current sequence
            # 3. requeue the rest of the messages
            self.sendqueue = []
            self.resend = False
        if len(self.sendqueue) == 0:
            # we need to fill it
            for local_seq_id in range(self.MAX_QUEUE_SIZE):
                sequence_id = self.expected_sequence + local_seq_id
                if sequence_id in self.sequence_queue:
                    self.sendqueue.append(self.sequence_queue[sequence_id])
                else:
                    break  # stop when we tested a nonexistant sequence
            # send the queue
            for i in range(len(self.sendqueue)):
                msg = self.sendqueue[i]
                self.data_handler.external_send_queue.append(
                    (msg, asyncio.Future(), time.monotonic())
                )
        else:
            # sendqueue is still full, so we need to wait until accepted
            ...
        # await asyncio.sleep(0.5)

    def do_not_wait_for_sequence(self, sequence: int) -> bool:
        if sequence in self.sequence_queue:
            return self.do_not_wait_for_msg(self.sequence_queue[sequence])
        return False

    def do_not_wait_for_msg(self, msg) -> bool:
        if msg.WhichOneof("message_type") == "configuration":
            if msg.configuration.execute_reset:
                self.sendqueue.insert(0, None)  # it will pop(0)
                self.accept_acknowledge(msg.sequence, True)
                return True  # do not wait for resets
        return False

    def get_success_for_sequence_list(self, sequence_list: list[int]) -> dict[bool]:
        return {
            sequence: (
                False
                if sequence not in self.sequence_queue
                else self.sequence_success[sequence]
            )
            for sequence in sequence_list
        }

    async def wait_for_sequence(self, sequence) -> None:
        def completed_future(result):
            fut = asyncio.Future()
            fut.set_result(result)
            return fut

        self.wait_queue[sequence] = asyncio.Future()
        logging.debug(f">>>start waiting for sequence {sequence}")
        if self.do_not_wait_for_sequence(sequence):
            logging.debug(f"dont wait for sequence {sequence}, but sleep")
            await asyncio.sleep(0.5)  # wait half a second for a reboot
        else:
            await self.wait_queue[sequence]
        logging.debug(f"<<<done  waiting for sequence {sequence}")
        return self.wait_queue.get(sequence, completed_future(True)).result()


class DataHandler:
    def __init__(self, reader, writer, debug: bool = False):
        self.reader = reader
        self.writer = writer
        self.debug = debug
        self.handlers: list[MessageHandler] = [self.keepalive_handler, self.ack_handler, self.payload_handler]

        self.msgs = []
        self.results = {}
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

        self.packet_order_sender = PacketOrderSender(self)

    async def ack_handler(self, msg, writer) -> bool:
        if msg.WhichOneof("message_type") == "ack":
            try:
                # print("ACK", msg.sequence)
                seq = msg.sequence
                success = msg.ack.success
                assert self.packet_order_sender.accept_acknowledge(seq, success), (
                    f"cannot accept seq {seq} (with: "
                    f"expecting {self.packet_order_sender.expected_sequence}, "
                    f"enqueued {self.packet_order_sender.enqueued_sequence})"
                )
            except Exception:
                import traceback

                traceback.print_exc()
            finally:
                return True
        return False

    async def payload_handler(self, msg, writer) -> bool:
        if msg.WhichOneof("message_type") == "payload":
            try:
                # this may be a result
                for setmem in msg.payload.memory:
                    # device is delivering device read answers
                    if msg.sequence not in self.results:
                        self.results[msg.sequence] = []
                    self.results[msg.sequence].append({"addr":setmem.address, "data":setmem.data})
            except Exception:
                import traceback

                traceback.print_exc()
            finally:
                return True
        return False

    async def keepalive_handler(self, msg, writer) -> bool:
        if msg.WhichOneof("message_type") == "keepalive":
            try:
                for warning_id, warning_text in inverted_warning_flags.items():
                    if msg.keepalive.warning & warning_id:
                        logging.warning(
                            f"[{msg.keepalive.ticks:06d}] Device: {warning_text}"
                        )
                # logging.debug(
                #     f"[ALIVE n={msg.keepalive.next_tick}] {msg.keepalive.ticks}"
                #     f" [I: {msg.keepalive.inqueue_size:03d} "
                #     f"/ O: {msg.keepalive.outqueue_size:03d}]"
                # )
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
            return True
        return False

    async def default_handle_msg(self, msg, writer) -> bool:
        pass
        print(
            f"[? {msg.WhichOneof('message_type')} ]{str(msg)[:600]} "
            f"'{msg.SerializeToString()}'"
        )

    async def send_msgs(
        self, msgs: list["main_pb2.ProtocolMessage"], group_timeout: int = 4.0
    ):
        # send out the sequence, but only check the last message of the list,
        # and only return true if all sequences returned success=True
        sequences = [self.packet_order_sender.enqueue(msg) for msg in msgs]
        last_sequence = sequences[-1]
        try:
            futures = {
                sequence: self.packet_order_sender.wait_for_sequence(sequence)
                for sequence in [last_sequence]
            }

            last_future = futures[last_sequence]
            await last_future
            success_per_sequence = (
                self.packet_order_sender.get_success_for_sequence_list(sequences)
            )
            all_success = all(
                success_per_sequence.values()
            )  # only true if all are successful
            if not all_success:
                unsuccessful_sequences = [
                    seq for seq, success in success_per_sequence.items() if not success
                ]
                logging.info(f"sequences {unsuccessful_sequences} did not return true")
            if all_success and len(self.results) > 0:
                results = []
                for sequence in sequences:
                    if sequence in self.results:
                        results.extend(self.results[sequence])
                        del self.results[sequence]
                return results
            return all_success
        except TimeoutError:
            raise Exception("Timeout on wait for ack packet")

    async def handle_sendqueue(self, writer) -> None:
        # sending
        await self.packet_order_sender.work()
        if self.debug:
            ret1 = writer.write(self.debug_data)
            ret2 = await writer.drain()
        elif len(self.external_send_queue) > 0:
            for i_msg in range(len(self.external_send_queue)):
                msg, future, inqueue_timestamp = self.external_send_queue.pop(0)
                # print(f"time_in_queue: {time.monotonic() - inqueue_timestamp:2.2f}")
                if msg:
                    if msg is None:
                        future.set_result(16)
                        return

                    # carefully craft the crc of the message to be 0x0
                    msg.checksum = recalc_crc(msg)
                    serdata = msg.SerializeToString()
                    if not msg.keepalive.ticks:
                        msgstr = str(msg).replace("\n", " ")
                        logging.debug(
                            f"sent msg [len={len(serdata)}, "
                            f"seq={msg.sequence}, msg{msgstr}]"
                        )
                    if len(serdata) > BUFFER_COMPARE_SIZE:
                        future.set_result(32)
                        # we dont want to overload the smol buffer
                        # (cant increase it because DMA)
                        return
                    # run length encoding
                    ret0 = writer.write(  # noqa F841
                        struct.pack("<H", len(serdata) * 8)
                    )  # in bits, to check if we received garbage
                    ret1 = writer.write(serdata)  # noqa F841
            ret2 = await writer.drain()  # noqa F841
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
            msgs_str = " | ".join(
                str(msg).replace("\n", " ").replace("\r", "") for msg in self.msgs
            )
            logging.debug(f"received {len(self.msgs)} messages: [{msgs_str}")
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
            # raw_msg with RLE
            raw_msg: bytes = data[: new_pos + length]
            data_msg: bytes = raw_msg[new_pos:]
            crc_val: bytes = data[new_pos + length : new_pos + length + 1]
            assert len(crc_val) and crc(raw_msg) == crc_val[0], "crc is wrong"
            ret.ParseFromString(data_msg)
            assert ret.version in range(10), "version is not viable"
        except google.protobuf.message.DecodeError as ex:
            logging.info("decode error")
            raise Exception() from ex
        except AssertionError as ae:
            # logging.error("crc error") or logging.error("version error")
            raise Exception() from ae
        return ret

    def search_protobuf_messages(
        self, datastream: bytes, discard_limit: int = 100
    ) -> tuple[list, bytes]:
        messages = []
        loop_while = len(datastream) > 0  # True, but does not allow always looping
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
                            # logging.warning(
                            # f"skipped data: {datastream[0:startindex]}")
                            pass
                        # Update the datastream to point to the next message
                        # + 1 for crc and + 1 for the length
                        datastream = datastream[
                            1 + startindex + message.ByteSize() + 1 :
                        ]
                        break
                except ValueError:
                    continue
                except Exception:
                    # If parsing fails, continue searching from the next position
                    # print(f"startindex {startindex} did not work", type(e), e)
                    ...
            if len(datastream) == 0:
                break  # read everything
        return messages, datastream

    async def find_message_handler(self, msg, writer) -> bool:
        for handler in self.handlers:
            try:
                if await handler(msg, writer):
                    return True
            except Exception:
                ...
        return await self.default_handle_msg(msg, writer)


async def await_closing_handle_serial(
    data_handler: DataHandler, reason_to_exit: str, writer: asyncio.StreamWriter
):
    if data_handler:
        # before we exit, we need to cancel all sendqueue futures
        for msg, future, timestamp in data_handler.external_send_queue:
            future.cancel(reason_to_exit)
    # we also want to kill the keepalive timer
    if data_handler:
        try:
            if data_handler.keepalive_timer:
                await data_handler.keepalive_timer.close_tasks()
            data_handler.keepalive_timer = None
        except Exception:
            traceback.print_exc()
    # and close the writer
    if writer:
        writer.close()
        await writer.wait_closed()
    await asyncio.sleep(0.5)


class CompatibilityAioserialWriter:
    def __init__(self, aioserial_instance):
        self.aioserial_instance = aioserial_instance
        self.data = b""

    def write(self, data):
        self.data += data

    def close(self):
        self.aioserial_instance.close()

    async def wait_closed(self):
        # we need to kill all worker threads of aioserial_instance
        executors = (
            [
                self.aioserial_instance._cancel_read_executor,
                self.aioserial_instance._read_executor,
                self.aioserial_instance._cancel_write_executor,
                self.aioserial_instance._write_executor,
            ]
            if isinstance(self.aioserial_instance, aioserial.AioSerial)
            else []
        )
        for executor in executors:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                import traceback

                traceback.print_exc()
        # serial should be closed

    async def drain(self):
        ret = await self.aioserial_instance.write_async(self.data)
        logging.debug(f"###      wrote {len(self.data)}")
        self.data = b""
        return ret


class CompatibilityAioserialReader:
    def __init__(self, aioserial_instance):
        self.aioserial_instance = aioserial_instance
        self.emit_every_threshold = 10
        self.consequtive_lengths = []

    async def read(self, size):
        # logging.debug(f"### bfore read {self.aioserial_instance.in_waiting}")
        try:
            ret = await self.aioserial_instance.read_async(
                self.aioserial_instance.in_waiting
            )
        except aioserial.serialutil.SerialException as serex:
            # maybe this is a multiple_access / disconnect?
            ret = b""
            if "readiness" in str(serex):
                # phew, we can ignore (but its a device reset *blush*)
                ...
            else:
                raise serex  # else we dont know
        # log every x lengths, if everything is 0, do emit
        self.consequtive_lengths.append(len(ret))
        if (
            sum(self.consequtive_lengths) == 0
            and (len(self.consequtive_lengths)) == self.emit_every_threshold
        ):  # we have only 0
            logging.debug(f"### after read {len(ret)}")
            self.consequtive_lengths = []

        return ret


async def open_serial_connection(*, loop=None, limit=None, **kwargs):
    """wrapper for aioserial with interface of serial_asyncio.open_serial_connection"""

    url = kwargs.get("url")
    baudrate = kwargs.get("baudrate")

    # check if the URL is /dev/virtual..... else use the compatibility serial
    # "VirtualDevice"
    if "virtual" in url:
        aioserial_instance = FullDevice(crc) if "Device" in url else MeasureDevice()
    else:
        aioserial_instance: aioserial.AioSerial = aioserial.AioSerial(
            port=url,
            baudrate=baudrate,
        )
        aioserial_instance.set_low_latency_mode(True)

    return CompatibilityAioserialReader(
        aioserial_instance
    ), CompatibilityAioserialWriter(aioserial_instance)


async def handle_write(data_handler, writer):
    try:
        await data_handler.handle_sendqueue(writer)
    except Exception:
        traceback.print_exc()


async def handle_read(data_handler, writer, timeout):
    msg = await data_handler.next_msg(timeout=timeout)
    if msg:
        # logging.debug("message found")
        try:
            await data_handler.find_message_handler(msg, writer)
        except Exception:
            traceback.print_exc()


async def handle_serial(
    commands: Commands,
    tty: str,
    debug: bool = False,
    timeout: float = 1,  # noqa: ASYNC109
    open_serial_connection_patchable=open_serial_connection,
    closed_future: asyncio.Future = None,
):
    global data
    reason_to_exit = ""
    data_handler = None
    writer = None
    try:
        reader, writer = await open_serial_connection_patchable(
            url=tty,
            baudrate=1_500_000,
            timeout=0.000,
            write_timeout=0.001,
        )
        data_handler = DataHandler(reader, writer, debug=debug)
        commands._register(data_handler)
        while not commands.set_exit_request:
            try:
                data_handler.keepalive_timer.check_and_raise()
                read_res, write_res = await asyncio.gather(
                    handle_read(data_handler, writer, timeout),
                    handle_write(data_handler, writer),
                )
            except TimeoutError as timeoutErr:
                if "Keepalive-Timer" in str(timeoutErr):
                    print("Timeout:", str(timeoutErr))
                    break
        if commands.set_exit_request:
            reason_to_exit = "exit requested"
    except Exception as loop_breaking_exception:
        traceback.print_exc()
        reason_to_exit = str(loop_breaking_exception)
    finally:
        with contextlib.suppress(RuntimeError):
            await await_closing_handle_serial(data_handler, reason_to_exit, writer)
    if closed_future:
        closed_future.set_result(reason_to_exit)
