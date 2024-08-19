import asyncio
import logging
import platform

import click

from netdeployonnx.devices.max78000.device_transport.cli import cli
from netdeployonnx.devices.max78000.device_transport.commands import Commands
from netdeployonnx.devices.max78000.device_transport.serialhandler import handle_serial


async def asyncmain(args, tty, debug, timeout):
    commands = Commands()
    funcs = []
    funcs += [handle_serial(commands, tty, debug, timeout)]
    funcs += [cli(commands, debug, timeout)]
    await asyncio.gather(*funcs)


# Configure logging to output to stdout and file 'debug.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('debug.log')
    ],
)


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
