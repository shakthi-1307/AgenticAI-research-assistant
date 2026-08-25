import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS


client = Groq(
    api_key=GROQ_API_KEY
)


MAX_RESEARCH_STEPS = 3


def research_task(task):

    messages = [
        {
            "role": "system",
            "content": """
You are a research worker.

Your job is to complete ONE research task.

Use web_search when external information
is required.

Use fetch_page when you need more detail
from a source.

Do not perform unrelated tasks.

Stop once you have enough information.

Return a concise factual result.
""",
        },
        {
            "role": "user",
            "content": (
                "Research task:\n"
                + task["task"]
            ),
        },
    ]

    for step in range(
        MAX_RESEARCH_STEPS
    ):

        print(
            f"Research worker step "
            f"{step + 1}"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1200,
        )

        message = response.choices[0].message

        messages.append(message)

        # -------------------------------------
        # Worker finished
        # -------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content,
            }

        # -------------------------------------
        # Execute worker tools
        # -------------------------------------

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(
                f"Research worker tool: "
                f"{name}"
            )

            # Import here to keep the worker
            # independent from the main executor.
            from .tools import execute_tool

            result = execute_tool(
                name,
                arguments
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "name": name,
                    "content": json.dumps(
                        result
                    ),
                }
            )

    return {
        "status": "failed",
        "answer": (
            "Research worker reached "
            "its maximum steps."
        ),
    }