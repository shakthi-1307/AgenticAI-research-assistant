import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool


client = Groq(api_key=GROQ_API_KEY)


def research(question: str) -> str:

    messages = [
        {
            "role": "system",
            "content": """
You are an autonomous research agent.

Research the user's question before answering.

Use web_search to find information.
Use fetch_page when you need to inspect a source.

You can call tools multiple times.

Prefer reliable and primary sources.
Compare information when sources disagree.
Do not invent facts or sources.

When you have enough evidence, stop using tools
and provide the final answer with source URLs.
""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for _ in range(10):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = execute_tool(
                name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "The research agent reached its maximum number of steps."
