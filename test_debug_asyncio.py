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


async def debug_queue():
    while True:
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        print(f"Current tasks in the execution queue ({len(tasks)}):")
        for task in tasks:
            print(f"- {task}")
        await asyncio.sleep(0.1)


async def main():
    # Add your async functions here
    async def my_async_function():
        await asyncio.sleep(2)
        print("my_async_function completed")

    asyncio.create_task(my_async_function())
    asyncio.create_task(my_async_function())
    asyncio.create_task(my_async_function())

    # Start the debug coroutine
    debug_task = asyncio.create_task(debug_queue())

    # Wait for the debug coroutine to finish
    await debug_task


if __name__ == "__main__":
    asyncio.run(main())
