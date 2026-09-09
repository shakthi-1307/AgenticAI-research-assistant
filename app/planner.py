import json

from .llm import call_llm


def create_plan(question):

    system_prompt = """
You are a task planner for an Agentic AI system.

Break the user's request into executable tasks.

Available task types:

RESEARCH
{
  "type": "research",
  "task": "research question",
  "depends_on": []
}

WEATHER
{
  "type": "weather",
  "task": "get weather",
  "city": "Chennai",
  "depends_on": []
}

CALCULATION
{
  "type": "calculation",
  "task": "perform calculation",
  "expression": "10 / 2",
  "depends_on": []
}

DEPENDENCIES:

depends_on contains task indexes.

Indexes start at 0.

Example:

Task 0: Find India's population.
Task 1: Find China's population.
Task 2: Compare India's and China's populations.
Task 2: "depends_on": [0, 1]

Task 3: Calculate percentage difference using populations
from Tasks 0 and 1.
Task 3: "depends_on": [0, 1]

IMPORTANT:

- Independent tasks should use [].
- A task should depend on another task when it needs
  that task's result.
- Only reference earlier task indexes.
- Keep tasks simple and executable.
- Do not create unnecessary tasks.
- Return ONLY JSON.
- Do not use markdown.
- Do not include explanations outside JSON.

The output MUST have exactly:

{
  "tasks": [
    {
      "type": "research",
      "task": "...",
      "depends_on": []
    }
  ]
}
"""

    message = call_llm(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_object"
        },
        max_tokens=800
    )

    raw_content = message.content

    if not raw_content:
        raise ValueError(
            "Planner returned empty output."
        )

    # ---------------------------------------------------------
    # Parse planner JSON
    # ---------------------------------------------------------

    try:

        data = json.loads(raw_content)

    except json.JSONDecodeError as error:

        print("Planner returned invalid JSON:")
        print(raw_content)

        raise ValueError(
            f"Planner JSON parsing failed: {error}"
        )

    # ---------------------------------------------------------
    # Validate tasks
    # ---------------------------------------------------------

    if "tasks" not in data:

        raise ValueError(
            "Planner response does not contain 'tasks'."
        )

    if not isinstance(data["tasks"], list):

        raise ValueError(
            "Planner 'tasks' must be a list."
        )

    tasks = []

    for index, task in enumerate(data["tasks"]):

        if not isinstance(task, dict):

            raise ValueError(
                f"Task {index} is not a valid object."
            )

        if "type" not in task:

            raise ValueError(
                f"Task {index} has no type."
            )

        if "task" not in task:

            raise ValueError(
                f"Task {index} has no task description."
            )

        task["depends_on"] = task.get(
            "depends_on",
            []
        )

        if not isinstance(
            task["depends_on"],
            list
        ):

            raise ValueError(
                f"Task {index} dependencies must be a list."
            )

        # -----------------------------------------------------
        # Validate dependency indexes
        # -----------------------------------------------------

        for dependency in task["depends_on"]:

            if not isinstance(
                dependency,
                int
            ):

                raise ValueError(
                    f"Task {index} has an invalid "
                    f"dependency index."
                )

            if dependency < 0:

                raise ValueError(
                    f"Task {index} has a negative "
                    f"dependency index."
                )

            if dependency >= index:

                raise ValueError(
                    f"Task {index} depends on task "
                    f"{dependency}, but dependencies "
                    f"must reference earlier tasks."
                )

        # -----------------------------------------------------
        # Runtime state
        # -----------------------------------------------------

        task["status"] = "pending"
        task["retries"] = 0
        task["max_retries"] = 2

        tasks.append(task)

    return tasks


def get_ready_tasks(plan):

    ready = []

    for task in plan:

        if task["status"] not in (
            "pending",
            "failed"
        ):
            continue

        # Don't retry forever
        if (
            task["status"] == "failed"
            and task.get("retries", 0)
            >= task.get("max_retries", 2)
        ):
            continue

        dependencies = task.get(
            "depends_on",
            []
        )

        dependencies_complete = all(
            plan[dependency]["status"] == "complete"
            for dependency in dependencies
        )

        if dependencies_complete:

            ready.append(task)

    return ready


def all_tasks_complete(plan):

    return all(
        task["status"] == "complete"
        for task in plan
    )