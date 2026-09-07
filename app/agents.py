import json
from dataclasses import dataclass, field

from groq import Groq

from .context import build_context
from .config import GROQ_API_KEY, MODEL
from .executor import execute_ready_tasks
from .planner import (
    create_plan,
    get_ready_tasks,
    all_tasks_complete,
)


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5


@dataclass
class AgentState:

    messages: list = field(default_factory=list)

    plan: list = field(default_factory=list)

    results: list = field(default_factory=list)

    iteration: int = 0

    final_answer: str | None = None


def save_result(state, result):

    state.results.append(
        {
            "task": result["task"]["task"],
            "type": result["task"]["type"],
            "result": result["result"],
        }
    )


def get_working_memory(state):

    return state.results


def build_final_answer(state):

    context = build_context(state)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Answer the user's question using the
provided agent context.

Do not call tools.

Do not invent information.

Be concise and factual.
""",
            },
            {
                "role": "user",
                "content": (
                    f"Agent context:\n\n{context}"
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

    # ---------------------------------------------
    # Create task plan
    # ---------------------------------------------

    state.plan = create_plan(question)

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

    # ---------------------------------------------
    # Main agent loop
    # ---------------------------------------------

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        print(
            f"\n--- Agent iteration "
            f"{state.iteration} ---"
        )

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"Ready tasks: {len(ready_tasks)}"
        )

        if not ready_tasks:
            break

        # -----------------------------------------
        # Execute ready tasks
        # -----------------------------------------

        results = await execute_ready_tasks(
            ready_tasks
        )

        # -----------------------------------------
        # Process task results
        # -----------------------------------------

        for item in results:

            task = item["task"]

            result = item["result"]

            # -------------------------------------
            # Determine success
            # -------------------------------------

            if task["type"] == "research":

                success = (
                    isinstance(result, dict)
                    and result.get("status")
                    == "complete"
                )

            else:

                success = (
                    isinstance(result, dict)
                    and "error" not in result
                )

            # -------------------------------------
            # Successful task
            # -------------------------------------

            if success:

                task["status"] = "complete"

                print(
                    f"Task completed: "
                    f"{task['task']}"
                )

            # -------------------------------------
            # Failed task
            # -------------------------------------

            else:

                task["status"] = "failed"

                task["retries"] = (
                    task.get("retries", 0) + 1
                )

                print(
                    f"Task failed: "
                    f"{task['task']}"
                )

                print(
                    f"Retry count: "
                    f"{task['retries']}/"
                    f"{task.get('max_retries', 2)}"
                )

            save_result(
                state,
                item,
            )

        # -----------------------------------------
        # Show working memory
        # -----------------------------------------

        print("\nWorking memory:")

        print(
            json.dumps(
                state.results,
                indent=2,
            )
        )

        # -----------------------------------------
        # Show updated plan
        # -----------------------------------------

        print("\nUpdated plan:")

        for index, task in enumerate(
            state.plan,
            start=1,
        ):

            print(
                f"{index}. "
                f"{task['task']} "
                f"[{task['status']}] "
                f"(retries: "
                f"{task.get('retries', 0)})"
            )

        # -----------------------------------------
        # Stop if everything completed
        # -----------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            break

    # ---------------------------------------------
    # Generate final answer
    # ---------------------------------------------

    if all_tasks_complete(
        state.plan
    ):

        state.final_answer = (
            build_final_answer(state)
        )

        return state.final_answer

    return (
        "I could not complete the research."
    )