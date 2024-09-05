import asyncio
from unittest import mock

import pytest
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from netdeployonnx.devices.max78000.device_transport import serialhandler
from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.protobuffers import main_pb2

try:
    from itertools import batched
except ImportError:
    from itertools import islice

    def batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def open_serial_connection_virtual_device():
    import serial_asyncio

    original_func = serial_asyncio.open_serial_connection

    def wrap_read(func):
        async def read(*args, **kwargs):
            print("R:", args, kwargs, end=" ->")
            r = await func(*args, **kwargs)
            print(r)
            return r

        return read

    def wrap_write(func):
        def write(*args, **kwargs):
            print("W:", args, kwargs)
            return func(*args, **kwargs)

        return write

    async def return_virtual_dev(url, *args, **kwargs):
        reader = mock.Mock(asyncio.StreamReader)
        writer = mock.Mock(asyncio.StreamWriter)
        print("open", url, args, kwargs)
        r, w = await original_func(*args, url=url, **kwargs)
        reader.read = wrap_read(r.read)
        writer.write = wrap_write(w.write)
        writer.drain = w.drain
        return reader, writer

    return return_virtual_dev


@pytest.mark.asyncio
async def test_cli_connect(event_loop, open_serial_connection_virtual_device):
    with mock.patch(
        "netdeployonnx.devices.max78000.device_transport.serialhandler.open_serial_connection",
        open_serial_connection_virtual_device,
    ) as mock_open_serial_connection:  # noqa: F841
        # we have at first a task
        commands = Commands()
        task = event_loop.create_task(
            serialhandler.handle_serial(
                commands,
                tty="/dev/ttyUSB0",
                timeout=4,
                open_serial_connection_patchable=open_serial_connection_virtual_device,
            )
        )
        # now we have the task, now we need to send messages
        messages_per_stage = [
            (
                "stage1",
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                    for i in range(1000)
                ],
            ),
            (
                "stage2",
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                    for i in range(1000)
                ],
            ),
            (
                "stage3",
                [
                    main_pb2.ProtocolMessage(
                        version=2, action=main_pb2.Action(execute_measurement="NONE")
                    )
                    for i in range(1000)
                ],
            ),
        ]

        async def send_task():
            tasks = {}
            await asyncio.sleep(0)
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                for stagename, stage_messages in messages_per_stage:
                    tasks[stagename] = progress.add_task(stagename, total=1)
                for stagename, stage_messages in messages_per_stage:
                    for stagemsg in stage_messages:
                        messages = commands.split_message(stagemsg)
                        progress.reset(tasks[stagename], total=len(messages))
                        batchsize = 10
                        for batch in batched(enumerate(messages), batchsize):
                            await commands.send_batch(
                                submessage for index_submessage, submessage in batch
                            )  # these can throw a CancelledError
                            progress.advance(tasks[stagename], len(batch))

    await asyncio.gather(task, send_task())
