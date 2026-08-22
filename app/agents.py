import json
from dataclasses import dataclass, field

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5


@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None


def research(question: str) -> str:

    state = AgentState(
        messages=[
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
    )

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        print(f"\n--- Agent iteration {state.iteration} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=state.messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1500,
        )

        message = response.choices[0].message

        # Add the assistant's response to conversation.
        state.messages.append(message)

        # No tool call means the agent is finished.
        if not message.tool_calls:
            state.final_answer = message.content
            break

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

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                }
            )

    if state.final_answer:
        return state.final_answer

    return (
        "I could not complete the research within "
        "the maximum number of research steps."
    )