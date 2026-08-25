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
Break the user's request into the smallest
number of concrete tasks required to answer it.

Also identify dependencies.

A task depends on another task when it needs
the result of that task.

Return ONLY valid JSON.

Format:

{
    "tasks": [
        {
            "task": "task description",
            "depends_on": []
        }
    ]
}

Example:

{
    "tasks": [
        {
            "task": "Search for an article",
            "depends_on": []
        },
        {
            "task": "Fetch the article",
            "depends_on": [
                "Search for an article"
            ]
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
        max_tokens=700,
    )

    data = json.loads(
        response.choices[0].message.content
    )

    return [
        {
            "task": item["task"],
            "status": "pending",
            "depends_on": item.get(
                "depends_on",
                []
            ),
        }
        for item in data["tasks"]
    ]


def get_ready_tasks(plan):

    ready_tasks = []

    for item in plan:

        if item["status"] != "pending":
            continue

        dependencies = item["depends_on"]

        dependencies_complete = all(
            any(
                other["task"] == dependency
                and other["status"] == "complete"
                for other in plan
            )
            for dependency in dependencies
        )

        if dependencies_complete:
            ready_tasks.append(item)

    return ready_tasks


def all_tasks_complete(plan):

    if not plan:
        return False

    return all(
        item["status"] == "complete"
        for item in plan
    )