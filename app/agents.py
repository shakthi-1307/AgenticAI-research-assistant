import json
from dataclasses import dataclass, field

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .executor import execute_tool_calls
from .planner import (
    create_plan,
    get_ready_tasks,
    all_tasks_complete,
)
from .tools import TOOLS


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5


@dataclass
class AgentState:

    messages: list = field(
        default_factory=list
    )

    iteration: int = 0

    final_answer: str | None = None

    plan: list = field(
        default_factory=list
    )


def call_llm(messages):

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1500,
    )

    return response.choices[0].message


def update_plan(
    state,
    tool_results,
):

    for item in state.plan:

        task = item["task"].lower()

        for tool_result in tool_results:

            tool_name = tool_result["name"]
            result = tool_result["result"]

            if (
                isinstance(result, dict)
                and "error" in result
            ):
                continue

            if (
                tool_name == "get_weather"
                and "weather" in task
            ):
                item["status"] = "complete"

            elif (
                tool_name == "calculator"
                and (
                    "calcul" in task
                    or "multiply" in task
                    or "product" in task
                )
            ):
                item["status"] = "complete"


def build_final_answer(state):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Answer the user's original question using
the research results in the conversation.

Do not call any tools.

Be concise and factual.

If sources are available, include them.
""",
            },
            *state.messages,
        ],
        tool_choice="none",
        max_tokens=1500,
    )

    return response.choices[0].message.content


async def research(question: str) -> str:

    state = AgentState(
        messages=[
            {
                "role": "system",
                "content": """
You are an autonomous research assistant.

You are given a task plan.

Use tools only when necessary to gather
information required by the plan.

Do not repeat completed work.

Use:
- calculator for calculations
- get_weather for weather
- web_search for general research
- fetch_page for webpage details

Do not invent facts.
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    # -----------------------------------------
    # 1. CREATE PLAN
    # -----------------------------------------

    state.plan = create_plan(
        question
    )

    print("\nPlan:")

    for index, item in enumerate(
        state.plan,
        start=1,
    ):

        print(
            f"{index}. {item['task']} "
            f"[{item['status']}]"
        )

    # -----------------------------------------
    # 2. GIVE PLAN TO LLM
    # -----------------------------------------

    state.messages.append(
        {
            "role": "system",
            "content": (
                "Current plan:\n"
                + json.dumps(
                    state.plan,
                    indent=2,
                )
            ),
        }
    )

    # -----------------------------------------
    # 3. AGENT LOOP
    # -----------------------------------------

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"\nReady tasks: "
            f"{len(ready_tasks)}"
        )

        # -------------------------------------
        # ASK LLM WHAT INFORMATION IT NEEDS
        # -------------------------------------

        message = call_llm(
            state.messages
        )

        state.messages.append(
            message
        )

        # -------------------------------------
        # NO TOOL CALL
        # -------------------------------------

        if not message.tool_calls:

            state.final_answer = (
                message.content
            )

            break

        # -------------------------------------
        # EXECUTOR
        # -------------------------------------

        tool_results = (
            await execute_tool_calls(
                message.tool_calls
            )
        )

        # -------------------------------------
        # STORE TOOL RESULTS
        # -------------------------------------

        for item in tool_results:

            tool_call = item["tool_call"]

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "name": item["name"],
                    "content": json.dumps(
                        item["result"]
                    ),
                }
            )

        # -------------------------------------
        # UPDATE STATE
        # -------------------------------------

        update_plan(
            state,
            tool_results,
        )

        print("\nUpdated plan:")

        for index, item in enumerate(
            state.plan,
            start=1,
        ):

            print(
                f"{index}. {item['task']} "
                f"[{item['status']}]"
            )

        # -------------------------------------
        # CHECK COMPLETION
        # -------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            state.final_answer = (
                build_final_answer(
                    state
                )
            )

            break

    if state.final_answer:

        return state.final_answer

    return (
        "I could not complete the research "
        "within the maximum number of "
        "research steps."
    )