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
