import asyncio
import logging
import platform

import click

from netdeployonnx.devices.max78000.device_transport.cli import cli
from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.serialhandler import handle_serial


async def debug_queue():
    # inline:
    # [f"- {task.get_coro().__name__}: {task.get_coro().cr_frame.f_code.co_filename}:{task.get_coro().cr_frame.f_lineno}"[-50:] for task in [task for task in asyncio.all_tasks() if not task.done()]]  # noqa: E501
    while True:
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        logging.debug(f"Current tasks in the execution queue ({len(tasks)}):")
        for task in tasks:
            coro = task.get_coro()
            location = f"{coro.cr_frame.f_code.co_filename}:{coro.cr_frame.f_lineno}"
            logging.debug(f"- {coro.__name__}: {location[-50:]}")
        await asyncio.sleep(0.1)


async def asyncmain(args, tty, debug, timeout):
    commands = Commands()
    funcs = []
    funcs += [handle_serial(commands, tty, debug, timeout)]
    funcs += [cli(commands, debug, timeout)]
    # funcs += [debug_queue()]
    await asyncio.gather(*funcs)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Custom format for relativeCreated as 3.3f
        record.relativeCreated = (
            f"{record.relativeCreated / 1000:3.3f}"  # Convert ms to seconds
        )
        return super().format(record)


# Configure logging to output to stdout and file 'debug.log'
logging.basicConfig(
    level=logging.INFO,
    format="[%(relativeCreated)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('debug.log')
    ],
)
# Create a custom formatter
custom_formatter = CustomFormatter(
    "[%(relativeCreated)s] %(levelname)s: %(message)s",
)

# Apply the custom formatter to all handlers of the root logger
for handler in logging.getLogger().handlers:
    handler.setFormatter(custom_formatter)


@click.command()
@click.option("--hex", default=None, help="Hexadecimal value (default: 0205AB)")
@click.option("--test", default=None, help="produce testcase data")
@click.option("--debug", default=None, help="enable debug flag")
@click.option("--timeout", default=1, help="set timeout in seconds")
@click.option(
    "--tty",
    default=("COM2" if platform.system().lower() == "windows" else "/dev/ttyACM0"),
    help="set the tty",
)
def main(hex, test, debug, timeout, tty):
    if test:
        testnumber = int(test)
        if testnumber == 1:
            msg = asyncio.run(Commands().set_led(1))
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 2:
            msg = asyncio.run(Commands().set_speed("500"))
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 3:
            msg = asyncio.run(
                Commands().load_file(
                    "/home/vscode/_Masterarbeit_SS24/hannah-env/syshelpers/payload_stage3.pbenc"
                )
            )
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 4:
            msg = asyncio.run(
                Commands().load_file(
                    "/home/vscode/_Masterarbeit_SS24/hannah-env/syshelpers/payload_stage3.pbenc"
                )
            )
            msg.payload.registers.remove(msg.payload.registers[-1])
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 5:
            msg = asyncio.run(
                Commands().load_file(
                    "/home/vscode/_Masterarbeit_SS24/hannah-env/syshelpers/payload_stage3.pbenc"
                )
            )
            msg.payload.registers.remove(msg.payload.registers[-1])
            msg.payload.registers.remove(msg.payload.registers[-1])
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 6:
            msg = asyncio.run(
                Commands().load_file(
                    "/home/vscode/_Masterarbeit_SS24/hannah-env/syshelpers/payload_stage3.pbenc"
                )
            )
            msg.payload.registers.remove(msg.payload.registers[-1])
            msg.payload.registers.remove(msg.payload.registers[-1])
            msg.payload.registers.remove(msg.payload.registers[-1])
            msg.payload.registers.remove(msg.payload.registers[-1])
            lastreg = msg.payload.registers.pop()  # noqa F841
            from protobuffers import main_pb2

            msg.payload.memory.append(
                main_pb2.SetMemoryContent(address=0x50004000, data=b"\0\0\0\0")
            )
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
            print("".join([f"{b:02X}" for b in msg.SerializeToString()]))
        if testnumber == 7:
            from protobuffers import main_pb2

            msg = main_pb2.SetMemoryContent(address=0x50004000, data=b"\0\0\0\0")
            print(", ".join([f"0x{b:02X}" for b in msg.SerializeToString()]))
            print("".join([f"{b:02X}" for b in msg.SerializeToString()]))
        return
    if hex:
        from protobuffers import main_pb2

        hex_string = hex.replace(" ", "")

        data = bytes.fromhex(hex_string)
        msg = main_pb2.ProtocolMessage.FromString(data)
        print(msg)

        data = msg.SerializeToString()
        print(data)
    else:
        import sys

        asyncio.run(asyncmain(sys.argv, tty, debug=debug, timeout=timeout))


if __name__ == "__main__":
    main()
