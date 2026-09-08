import json

from groq import Groq

from .context import build_context
from .config import GROQ_API_KEY, MODEL
from .executor import execute_ready_tasks
from .planner import (
    create_plan,
    get_ready_tasks,
    all_tasks_complete
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint
)
from .state import AgentState


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5

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


async def research(question):

    checkpoint_state = load_checkpoint()

    # --------------------------------------------------
    # Check whether an existing checkpoint belongs
    # to this question
    # --------------------------------------------------

    if (
        checkpoint_state is not None
        and checkpoint_state.question.strip()
        == question.strip()
    ):

        state = checkpoint_state

        print(
            "\nResuming from checkpoint..."
        )

        print(
            f"Previous iteration: "
            f"{state.iteration}"
        )

        print(
            "\nRestored plan:"
        )

        for index, task in enumerate(
            state.plan
        ):

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

    else:

        # ----------------------------------------------
        # New question
        # ----------------------------------------------

        if checkpoint_state is not None:

            print(
                "\nExisting checkpoint belongs "
                "to a different question."
            )

            print(
                "Starting a new research session."
            )

            delete_checkpoint()

        state = AgentState(
            question=question,

            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        state.plan = create_plan(
            question
        )

        print(
            "\nPlan:"
        )

        for index, task in enumerate(
            state.plan
        ):

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

    # --------------------------------------------------
    # Execute tasks
    # --------------------------------------------------

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
            f"Ready tasks: "
            f"{len(ready_tasks)}"
        )

        if not ready_tasks:

            break

        results = await execute_ready_tasks(
            ready_tasks,
            state.plan
        )

        for item in results:

            task = item["task"]
            result = item["result"]

            # ------------------------------------------
            # Determine success
            # ------------------------------------------

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

            # ------------------------------------------
            # Successful task
            # ------------------------------------------

            if success:

                task["status"] = "complete"
                task["result"] = result

                print(
                    f"Task completed: "
                    f"{task['task']}"
                )

            # ------------------------------------------
            # Failed task
            # ------------------------------------------

            else:

                task["status"] = "failed"

                task["retries"] = (
                    task.get("retries", 0) + 1
                )

                print(
                    f"Task failed: "
                    f"{task['task']} "
                    f"(retry "
                    f"{task['retries']})"
                )

            save_result(
                state,
                item
            )

        # ------------------------------------------
        # Save checkpoint
        # ------------------------------------------

        save_checkpoint(
            state
        )

        print(
            "\nUpdated plan:"
        )

        for index, task in enumerate(
            state.plan
        ):

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

        # ------------------------------------------
        # Finished
        # ------------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            state.final_answer = (
                build_final_answer(state)
            )

            save_checkpoint(
                state
            )

            return state.final_answer

    return (
        "I could not complete the research."
    )