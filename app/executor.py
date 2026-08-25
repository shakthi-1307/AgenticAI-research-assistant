import asyncio


MAX_TOOL_RETRIES = 2


async def execute_tool_with_retry(
    tool_function,
    arguments,
):

    for attempt in range(
        MAX_TOOL_RETRIES + 1
    ):

        try:

            return await asyncio.to_thread(
                tool_function,
                **arguments,
            )

        except Exception as error:

            print(
                f"Tool failed "
                f"(attempt {attempt + 1}): "
                f"{error}"
            )

            if attempt == MAX_TOOL_RETRIES:
                return {
                    "error": str(error)
                }

            await asyncio.sleep(1)


async def execute_task(task):

    """
    Execute one planned task.
    """

    task_type = task["type"]

    if task_type == "weather":

        from .tools import get_weather

        print(
            f"Executing weather task: "
            f"{task['city']}"
        )

        result = await execute_tool_with_retry(
            get_weather,
            {
                "city": task["city"]
            },
        )

        return {
            "task": task,
            "result": result,
        }

    if task_type == "calculation":

        from .tools import calculator

        print(
            f"Executing calculation: "
            f"{task['expression']}"
        )

        result = await execute_tool_with_retry(
            calculator,
            {
                "expression": task["expression"]
            },
        )

        return {
            "task": task,
            "result": result,
        }

    return {
        "task": task,
        "result": {
            "error": (
                f"Unknown task type: "
                f"{task_type}"
            )
        },
    }


async def execute_ready_tasks(
    ready_tasks,
):

    print(
        f"\nExecuting "
        f"{len(ready_tasks)} "
        f"ready tasks in parallel."
    )

    tasks = [
        execute_task(task)
        for task in ready_tasks
    ]

    results = await asyncio.gather(
        *tasks
    )

    return results