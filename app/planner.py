import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL


client = Groq(api_key=GROQ_API_KEY)


def create_plan(question):

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": """
You are a task planner for an Agentic AI research system.

Your job is to break the user's request into
clear executable tasks.

Available task types:

1. research
   Used for finding and investigating information.

   Required fields:
   {
       "type": "research",
       "task": "what needs to be researched",
       "depends_on": []
   }


2. weather
   Used for getting current weather.

   Required fields:
   {
       "type": "weather",
       "task": "get weather for Chennai",
       "city": "Chennai",
       "depends_on": []
   }


3. calculation
   Used for mathematical calculations.

   Required fields:
   {
       "type": "calculation",
       "task": "calculate something",
       "expression": "10 / 2",
       "depends_on": []
   }


DEPENDENCIES:

Use "depends_on" when a task needs the
result of another task.

The value must contain the INDEX of the
previous task.

Example:

Task 0:
Research the population of Chennai.

Task 1:
Convert the Chennai population into millions.

Task 1 should have:

"depends_on": [0]


IMPORTANT:

- depends_on controls both execution order
  and information flow.
- A task should depend on another task if it
  needs information produced by that task.
- Independent tasks should have [].
- Do not create unnecessary dependencies.
- Dependencies must refer to earlier tasks.
- Use task indexes starting from 0.
- Break complex requests into logical steps.
- Keep tasks specific and executable.

Examples:

User:
"Compare the populations of India and China."

Possible plan:

{
    "tasks": [
        {
            "type": "research",
            "task": "Find the current population of India.",
            "depends_on": []
        },
        {
            "type": "research",
            "task": "Find the current population of China.",
            "depends_on": []
        }
    ]
}


User:
"Find India's population and calculate it in millions."

Possible plan:

{
    "tasks": [
        {
            "type": "research",
            "task": "Find India's current population.",
            "depends_on": []
        },
        {
            "type": "calculation",
            "task": "Convert India's population into millions.",
            "expression": "POPULATION / 1000000",
            "depends_on": [0]
        }
    ]
}


Return ONLY valid JSON.

The JSON must have this structure:

{
    "tasks": [
        {
            "type": "research | weather | calculation",
            "task": "...",
            "depends_on": []
        }
    ]
}
"""
            },

            {
                "role": "user",
                "content": question
            }
        ],

        response_format={
            "type": "json_object"
        },

        max_tokens=1000
    )

    data = json.loads(
        response.choices[0].message.content
    )

    tasks = []

    for task in data["tasks"]:

        task["status"] = "pending"
        task["retries"] = 0
        task["max_retries"] = 2

        # Make sure depends_on exists
        task["depends_on"] = task.get(
            "depends_on",
            []
        )

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