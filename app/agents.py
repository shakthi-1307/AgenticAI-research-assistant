import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5


def research(question: str) -> str:

    messages = [
        {
            "role": "system",
            "content": """
You are an autonomous research assistant.

Your job is to research the user's question and provide
a factual answer based on web sources.

Rules:

1. Use web_search when current or external information is needed.
2. Use fetch_page only when you need more detail from a specific source.
3. Keep research focused.
4. Do not repeatedly search the same thing.
5. Do not invent facts or sources.
6. Prefer authoritative sources.
7. Stop researching once you have enough evidence.
8. Give a concise final answer with source URLs.
""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for iteration in range(MAX_ITERATIONS):

        print(f"\n--- Agent iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1500,
        )

        message = response.choices[0].message

        # Add the assistant's response to conversation.
        messages.append(message)

        # No tool call means the agent is finished.
        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"Tool: {name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(
                name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                }
            )

    return (
        "I could not complete the research within "
        "the maximum number of research steps."
    )

