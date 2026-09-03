import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL


client = Groq(
    api_key=GROQ_API_KEY
)


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

research
weather
calculation

Use:

research
for questions requiring general knowledge
or web research.

Use:

weather
for current weather.

Use:

calculation
for mathematical calculations.

Return ONLY valid JSON.

Example:

{
    "tasks": [
        {
            "task": "Research what artificial intelligence is",
            "type": "research",
            "depends_on": []
        },
        {
            "task": "Calculate 2 multiplied by 3",
            "type": "calculation",
            "expression": "2 * 3",
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
        response_format={
            "type": "json_object"
        },
        max_tokens=800,
    )

    data = json.loads(response.choices[0].message.content)

    return [
        {
            **task,
            "status": "pending",
        }
        for task in data["tasks"]
    ]


def get_ready_tasks(plan):

    ready = []

    for index, task in enumerate(plan):

        if task["status"] not in (
            "pending",
            "failed",
        ):
            continue

        # Don't retry beyond limit
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
            plan[dependency]["status"]
            == "complete"
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