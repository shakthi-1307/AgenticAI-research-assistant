from .research_worker import research_task
from .tools import execute_tool


async def execute_worker(task):

    task_type = task["type"]

    # Research task
    if task_type == "research":

        return await research_task(task)

    # Calculation task
    if task_type == "calculation":

        return execute_tool(
            "calculator",
            {
                "expression": task["expression"]
            }
        )

    # Weather task
    if task_type == "weather":

        return execute_tool(
            "get_weather",
            {
                "city": task["city"]
            }
        )

    return {
        "error": f"Unknown task type: {task_type}"
    }