#!/usr/bin/env python
import asyncio

print("Concurrent I/O Operations")
async def io_operation(task_name, delay):
    await asyncio.sleep(delay)
    print(f"Task {task_name} completed")

async def main():
    tasks = [
        io_operation("A", 3),
        io_operation("B", 1),
        io_operation("C", 2)
    ]
    await asyncio.gather(*tasks)

asyncio.run(main())


print("Creating Your Own Awaitable")


class MyAwaitable:
    def __await__(self):
        """
        Generator function, since 'yield' is in it

        Generator functions are functions that, when called, return a generator
        object instead of immediately executing the entire function and
        returning a value.

        Laxy Evalutation - Instead of computing and storing all values at once,
        generators produce values one at a time, on demand.
        This makes generators highly memory-efficient, especially when dealing
        with infinite sequences, as they don't need to load the entire sequence
        into memory simultaneously.

        When a 'yield' statement is encountered during iteration, the
        function's execution is paused, the yield value is returned to caller,
        and function's state is saved (including local variables and the
        execution point).
        """
        yield
        return 73

async def main():
    result = await MyAwaitable()
    print(result)

asyncio.run(main())

print("Multiple Tasks")

async def foo():
    """
    This code creates two tasks and runs them concurrently, making your code more efficient.
    """
    await asyncio.sleep(2)
    print("Hello from foo!")
async def bar():
    await asyncio.sleep(1)
    print("Hello from bar!")
async def main():
    task1 = asyncio.create_task(foo())
    task2 = asyncio.create_task(bar())
    await task1
    await task2
asyncio.run(main())

print("Real-world us case")

import aiohttp
from bs4 import BeautifulSoup
"""
Imagine a scenario where you want to download multiple web pages concurrently and extract specific information.
Here, we fetch and parse multiple web pages concurrently and extract their titles
"""
async def fetch_and_parse_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            page_content = await response.text()
            soup = BeautifulSoup(page_content, "html.parser")
            title = soup.title.string
            return f"Title of {url}: {title}"
async def main():
    urls = [
        "https://example.com",
        "https://python.org",
        "https://google.com",
    ]
    tasks = [fetch_and_parse_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)
asyncio.run(main())
