# Python Async Best Practices

## Never Block the Event Loop
Blocking calls (time.sleep, requests.get) inside async functions freeze all coroutines.
Use asyncio.sleep and httpx (async HTTP) instead.

BAD: async def fetch(): time.sleep(1); requests.get(url)
GOOD: async def fetch(): await asyncio.sleep(1); await client.get(url)

## Use asyncio.gather for Concurrent Tasks
Run independent coroutines concurrently, not sequentially.
Sequential: result_a = await task_a(); result_b = await task_b()
Concurrent: result_a, result_b = await asyncio.gather(task_a(), task_b())

## Timeout Everything
Network calls can hang. Always wrap with asyncio.wait_for.

## Async Context Managers
Use async with for resources that need async setup/teardown.
