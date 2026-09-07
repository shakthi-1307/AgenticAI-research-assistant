import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL


client = Groq(api_key=GROQ_API_KEY)


def create_plan(question: str):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Break the user's request into concrete tasks.

For every task identify:

- task
- type
- required arguments
- dependencies

Supported task types:

1. research

Use research for questions requiring
general knowledge or web research.

Required arguments:
- task
- type
- depends_on

2. weather

Use weather when the user asks for
current weather.

Required arguments:
- task
- type
- city
- depends_on

3. calculation

Use calculation for mathematical calculations.

Required arguments:
- task
- type
- expression
- depends_on

IMPORTANT:

- Always include all required arguments.
- For weather tasks, ALWAYS include "city".
- For calculation tasks, ALWAYS include "expression".
- For research tasks, no additional argument is required.
- Use [] for depends_on when there are no dependencies.
- Return ONLY valid JSON.

Example:

{
    "tasks": [
        {
            "task": "Research the differences between AI and ML",
            "type": "research",
            "depends_on": []
        },
        {
            "task": "Get the current weather in Chennai",
            "type": "weather",
            "city": "Chennai",
            "depends_on": []
        }
    ]
}
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=800,
    )

    data = json.loads(
        response.choices[0].message.content
    )

    tasks = []

    for task in data["tasks"]:

        task["status"] = "pending"

        # Track how many times this task has been retried.
        task["retries"] = 0

        # Maximum number of retries after the first attempt.
        task["max_retries"] = 2

        tasks.append(task)

    return tasks


def get_ready_tasks(plan):

    ready = []

    for task in plan:

        # Only pending or retryable failed tasks
        # should be executed.
        if task["status"] not in (
            "pending",
            "failed",
        ):
            continue

        # Do not retry a task after it has exhausted
        # its retry limit.
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