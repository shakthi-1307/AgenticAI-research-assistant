from .research_worker import research_task
from .tools import execute_tool


async def execute_worker(task, dependency_results=None):

    dependency_results = dependency_results or []

    task_type = task["type"]

    if task_type == "research":

        return await research_task(
            task,
            dependency_results
        )

    if task_type == "calculation":

        return execute_tool(
            "calculator",
            {
                "expression": task["expression"]
            }
        )

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