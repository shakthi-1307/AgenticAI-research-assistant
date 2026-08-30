import json
from dataclasses import dataclass, field

from groq import Groq
from .executor import execute_ready_tasks
from .context import build_context
from .config import GROQ_API_KEY, MODEL
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

    """
    Parent Agent.

    Responsible for:

    1. Creating the plan
    2. Finding ready tasks
    3. Sending tasks to executor
    4. Collecting results
    5. Updating state
    6. Producing final answer
    """

    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    # -----------------------------------------
    # 1. CREATE PLAN
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
    # 2. EXECUTION LOOP
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
        # 3. EXECUTE READY TASKS
        # -------------------------------------

        results = await execute_ready_tasks(
            ready_tasks
        )

        # -------------------------------------
        # 4. PROCESS RESULTS
        # -------------------------------------

        for item in results:

            task = item["task"]
            result = item["result"]

            # Research worker result
            if task["type"] == "research":

                if (
                    isinstance(result, dict)
                    and result.get("status")
                    == "complete"
                ):
                    task["status"] = (
                        "complete"
                    )
                else:
                    task["status"] = "failed"

            # Calculation result
            else:

                if (
                    isinstance(result, dict)
                    and "error" not in result
                ):
                    task["status"] = (
                        "complete"
                    )
                else:
                    task["status"] = "failed"

            save_result(
                state,
                item
            )

        # -------------------------------------
        # 5. SHOW WORKING MEMORY
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
        # 6. SHOW UPDATED PLAN
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
        # 7. CHECK COMPLETION
        # -------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            break

    # -----------------------------------------
    # 8. FINAL ANSWER
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