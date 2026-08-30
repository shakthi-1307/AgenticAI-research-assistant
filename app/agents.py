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


client = Groq(
    api_key=GROQ_API_KEY
)

MAX_ITERATIONS = 5


@dataclass
class AgentState:

    # What the LLM sees as conversation/context
    messages: list = field(
        default_factory=list
    )

    # What the planner/runtime tracks
    plan: list = field(
        default_factory=list
    )

    # Working memory
    results: list = field(
        default_factory=list
    )

    # Runtime metadata
    iteration: int = 0

    # Final response
    final_answer: str | None = None


def save_result(state, result):
    """
    Store a completed task result
    in working memory.
    """

    state.results.append(
        {
            "task": result["task"]["task"],
            "type": result["task"]["type"],
            "result": result["result"],
        }
    )


def get_working_memory(state):
    """
    Return information collected
    during the current agent run.
    """

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
                    f"Agent context:\n\n"
                    f"{context}"
                ),
            },
        ],
        tool_choice="none",
        max_tokens=1500,
    )

    return response.choices[0].message.content


async def research(question: str):

    # -----------------------------------------
    # 1. CREATE AGENT STATE
    # -----------------------------------------

    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    # -----------------------------------------
    # 2. CREATE PLAN
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
    # 3. MAIN AGENT LOOP
    # -----------------------------------------

    while (
        state.iteration
        < MAX_ITERATIONS
    ):

        state.iteration += 1

        # -------------------------------------
        # Find tasks that can run
        # -------------------------------------

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"\nReady tasks: "
            f"{len(ready_tasks)}"
        )

        # -------------------------------------
        # No tasks available
        # -------------------------------------

        if not ready_tasks:
            break

        # -------------------------------------
        # Mark tasks as in progress
        # -------------------------------------

        for task in ready_tasks:

            task["status"] = "in_progress"

        # -------------------------------------
        # Execute ready tasks in parallel
        # -------------------------------------

        results = await execute_ready_tasks(
            ready_tasks
        )

        # -------------------------------------
        # Process worker results
        # -------------------------------------

        for item in results:

            task = item["task"]
            result = item["result"]

            # ---------------------------------
            # RESEARCH TASK
            # ---------------------------------

            if task["type"] == "research":

                if (
                    isinstance(result, dict)
                    and result.get("status")
                    == "complete"
                ):

                    task["status"] = "complete"

                else:

                    task["retries"] = (
                        task.get(
                            "retries",
                            0
                        ) + 1
                    )

                    print(
                        f"Task failed: "
                        f"{task['task']}"
                    )

                    print(
                        f"Retry count: "
                        f"{task['retries']}"
                    )

                    if (
                        task["retries"]
                        >= task.get(
                            "max_retries",
                            2
                        )
                    ):

                        task["status"] = (
                            "failed"
                        )

                    else:

                        task["status"] = (
                            "pending"
                        )

            # ---------------------------------
            # OTHER TASKS
            # ---------------------------------

            else:

                if (
                    isinstance(result, dict)
                    and "error" not in result
                ):

                    task["status"] = (
                        "complete"
                    )

                else:

                    task["retries"] = (
                        task.get(
                            "retries",
                            0
                        ) + 1
                    )

                    print(
                        f"Task failed: "
                        f"{task['task']}"
                    )

                    print(
                        f"Retry count: "
                        f"{task['retries']}"
                    )

                    if (
                        task["retries"]
                        >= task.get(
                            "max_retries",
                            2
                        )
                    ):

                        task["status"] = (
                            "failed"
                        )

                    else:

                        task["status"] = (
                            "pending"
                        )

            # ---------------------------------
            # SAVE RESULT
            # ---------------------------------

            save_result(
                state,
                item
            )

        # -------------------------------------
        # SHOW WORKING MEMORY
        # -------------------------------------

        print(
            "\nWorking memory:"
        )

        print(
            json.dumps(
                state.results,
                indent=2
            )
        )

        # -------------------------------------
        # SHOW UPDATED PLAN
        # -------------------------------------

        print(
            "\nUpdated plan:"
        )

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
        # CHECK COMPLETION
        # -------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            break

    # -----------------------------------------
    # 4. BUILD FINAL ANSWER
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

    # -----------------------------------------
    # 5. FAILURE
    # -----------------------------------------

    return (
        "I could not complete the research."
    )