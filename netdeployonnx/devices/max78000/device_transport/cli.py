import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from netdeployonnx.devices.max78000.device_transport.commands import Commands


async def cli(cmdobj: Commands, debug: bool = False, timeout: float = 1):
    """

    alternative with cmd2: https://github.com/python-cmd2/cmd2/issues/764#issuecomment-1086334377
    """

    session = PromptSession()
    commanddict = cmdobj.get_commands()
    options = {
        "debug": debug,
        "timeout": timeout,
    }
    while True:
        try:
            # if session==None:
            #     await commanddict["awx"]()
            #     break
            with patch_stdout():
                command = await session.prompt_async(">> ")
                # Add your command handling logic here
                cmd = command.strip()
                cmds = cmd.split(" ")
                basecmd, cmdargs = cmds[0], cmds[1:]
                if not basecmd:
                    continue
                elif basecmd in commanddict:
                    await commanddict[basecmd](*cmdargs, **options)
                else:
                    print("unknown command:", basecmd)
        except KeyboardInterrupt:
            print("Exiting...")
            break
    cmdobj.exit_request()


async def main(): ...


if __name__ == "__main__":
    asyncio.run(main())
