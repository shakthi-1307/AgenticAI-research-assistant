from .research_worker import research_task
from .tools import execute_tool


async def execute_worker(task):

    task_type = task["type"]

    if task_type == "research":

        return await research_task(task)

    if task_type == "calculation":

        return execute_tool(
            "calculator",
            {
                "expression": task["expression"]
            }
        )

    return {
        "error": f"Unknown task type: {task_type}"
    }