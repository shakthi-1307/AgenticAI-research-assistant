import asyncio

from .workers import execute_worker


async def execute_ready_tasks(tasks):

    print(
        f"\nExecuting {len(tasks)} ready tasks in parallel."
    )

    async def run_task(task):

        print(
            f"Executing {task['type']} task: "
            f"{task['task']}"
        )

        result = await execute_worker(task)

        return {
            "task": task,
            "result": result,
        }

    results = await asyncio.gather(
        *[
            run_task(task)
            for task in tasks
        ],
        return_exceptions=False,
    )

    return results