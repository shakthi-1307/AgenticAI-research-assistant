import asyncio
from .workers import execute_worker


def get_dependency_results(task, plan):
    """
    Collect the completed results of all tasks
    this task depends on.
    """

    dependency_results = []

    for dependency_index in task.get("depends_on", []):
        dependency_task = plan[dependency_index]

        dependency_results.append({
            "task": dependency_task["task"],
            "type": dependency_task["type"],
            "result": dependency_task.get("result")
        })

    return dependency_results


async def execute_ready_tasks(tasks, plan):

    print(f"\nExecuting {len(tasks)} ready tasks in parallel.")

    async def run_task(task):

        print(f"Executing {task['type']} task: {task['task']}")

        dependency_results = get_dependency_results(
            task,
            plan
        )

        if dependency_results:
            print(
                f"Task depends on {len(dependency_results)} "
                f"completed task(s)."
            )

        result = await execute_worker(
            task,
            dependency_results
        )

        return {
            "task": task,
            "result": result
        }

    results = await asyncio.gather(
        *[run_task(task) for task in tasks],
        return_exceptions=False
    )

    return results