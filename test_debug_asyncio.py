import asyncio
import time

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