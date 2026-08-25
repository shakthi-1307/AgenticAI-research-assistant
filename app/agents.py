import json
from dataclasses import dataclass, field

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .executor import execute_ready_tasks
from .planner import (
    create_plan,
    get_ready_tasks,
    all_tasks_complete,
)


client = Groq(
    api_key=GROQ_API_KEY
)

MAX_ITERATIONS = 5


@dataclass
class AgentState:

    messages: list = field(
        default_factory=list
    )

    iteration: int = 0

    plan: list = field(
        default_factory=list
    )

    results: list = field(
        default_factory=list
    )

    final_answer: str | None = None


def build_final_answer(state):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Answer the user's question using the
completed task results.

Be concise and factual.

Do not call tools.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": (
                            state.messages[0]
                            ["content"]
                        ),
                        "results": state.results,
                    },
                    indent=2,
                ),
            },
        ],
        tool_choice="none",
        max_tokens=1500,
    )

    return response.choices[0].message.content


async def research(question: str):

    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    # -----------------------------------------
    # PLAN
    # -----------------------------------------

    state.plan = create_plan(
        question
    )

    print("\nPlan:")

    for index, task in enumerate(
        state.plan,
        start=1,
    ):

        print(
            f"{index}. "
            f"{task['task']} "
            f"[{task['status']}]"
        )

    # -----------------------------------------
    # EXECUTION LOOP
    # -----------------------------------------

    while (
        state.iteration
        < MAX_ITERATIONS
    ):

        state.iteration += 1

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"\nReady tasks: "
            f"{len(ready_tasks)}"
        )

        if not ready_tasks:

            break

        # -------------------------------------
        # PARALLEL EXECUTION
        # -------------------------------------

        results = await execute_ready_tasks(
            ready_tasks
        )

        # -------------------------------------
        # SAVE RESULTS
        # -------------------------------------

        for item in results:

            task = item["task"]

            task["status"] = "complete"

            state.results.append(
                item
            )

        # -------------------------------------
        # SHOW STATE
        # -------------------------------------

        print("\nUpdated plan:")

        for index, task in enumerate(
            state.plan,
            start=1,
        ):

            print(
                f"{index}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

        # -------------------------------------
        # DONE?
        # -------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            break

    # -----------------------------------------
    # FINAL ANSWER
    # -----------------------------------------

    if all_tasks_complete(
        state.plan
    ):

        state.final_answer = (
            build_final_answer(
                state
            )
        )

        return state.final_answer

    return (
        "I could not complete the research."
    )