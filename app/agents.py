import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .context import build_context
from .executor import execute_ready_tasks
from .planner import (
    create_plan,
    get_ready_tasks,
    all_tasks_complete
)
from .checkpoint import (
    create_run_id,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint
)
from .state import AgentState


client = Groq(
    api_key=GROQ_API_KEY
)


MAX_ITERATIONS = 5


def save_result(state, result):

    state.results.append(
        {
            "task": result["task"]["task"],
            "type": result["task"]["type"],
            "result": result["result"]
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
You are the final answer generator for an
Agentic AI research system.

Answer the user's original question using
the information collected by the agent.

Rules:

- Use only the provided agent context.
- Do not call tools.
- Do not invent information.
- Prefer factual and relevant information.
- If the context contains estimates, clearly
  identify them as estimates.
- Keep the answer concise.
"""
            },

            {
                "role": "user",
                "content": (
                    f"Original question:\n"
                    f"{state.question}\n\n"
                    f"Agent context:\n\n"
                    f"{context}"
                )
            }
        ],

        tool_choice="none",

        max_tokens=1500
    )

    return response.choices[0].message.content


async def research(question, run_id=None):

    # --------------------------------------------------
    # Create a run ID if this is a new execution
    # --------------------------------------------------

    if run_id is None:

        run_id = create_run_id()

    print(
        f"\nRun ID: {run_id}"
    )

    # --------------------------------------------------
    # Try to load an existing checkpoint
    # --------------------------------------------------

    checkpoint_state = load_checkpoint(
        run_id
    )

    # --------------------------------------------------
    # Resume existing run
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

    # --------------------------------------------------
    # New run
    # --------------------------------------------------

    else:

        if checkpoint_state is not None:

            print(
                "\nCheckpoint belongs to "
                "a different question."
            )

            print(
                "Starting a new state."
            )

            delete_checkpoint(
                run_id
            )

        state = AgentState(
            question=question,

            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        # --------------------------------------------------
        # Create plan
        # --------------------------------------------------

        state.plan = create_plan(
            question
        )

        print(
            "\nPlan:"
        )

        for index, task in enumerate(
            state.plan
        ):

            dependencies = task.get(
                "depends_on",
                []
            )

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

            if dependencies:

                print(
                    f"   depends_on: "
                    f"{dependencies}"
                )

    # --------------------------------------------------
    # Main agent execution loop
    # --------------------------------------------------

    while (
        state.iteration
        < MAX_ITERATIONS
    ):

        state.iteration += 1

        print(
            f"\n--- Agent iteration "
            f"{state.iteration} ---"
        )

        # --------------------------------------------------
        # Find tasks whose dependencies are complete
        # --------------------------------------------------

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"Ready tasks: "
            f"{len(ready_tasks)}"
        )

        # --------------------------------------------------
        # No tasks available
        # --------------------------------------------------

        if not ready_tasks:

            print(
                "No ready tasks."
            )

            break

        # --------------------------------------------------
        # Execute independent tasks in parallel
        # --------------------------------------------------

        results = await execute_ready_tasks(
            ready_tasks,
            state.plan
        )

        # --------------------------------------------------
        # Process task results
        # --------------------------------------------------

        for item in results:

            task = item["task"]
            result = item["result"]

            # ----------------------------------------------
            # Determine whether the task succeeded
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Successful task
            # ----------------------------------------------

            if success:

                task["status"] = "complete"

                # Store the actual result inside
                # the plan so dependent tasks can
                # access it.

                task["result"] = result

                print(
                    f"Task completed: "
                    f"{task['task']}"
                )

            # ----------------------------------------------
            # Failed task
            # ----------------------------------------------

            else:

                task["status"] = "failed"

                task["retries"] = (
                    task.get("retries", 0)
                    + 1
                )

                print(
                    f"Task failed: "
                    f"{task['task']} "
                    f"(retry "
                    f"{task['retries']})"
                )

            # ----------------------------------------------
            # Add result to working memory
            # ----------------------------------------------

            save_result(
                state,
                item
            )

        # --------------------------------------------------
        # Save checkpoint after the iteration
        # --------------------------------------------------

        save_checkpoint(
            state,
            run_id
        )

        # --------------------------------------------------
        # Display working memory
        # --------------------------------------------------

        print(
            "\nWorking memory:"
        )

        print(
            json.dumps(
                get_working_memory(state),
                indent=2,
                ensure_ascii=False
            )
        )

        # --------------------------------------------------
        # Display updated plan
        # --------------------------------------------------

        print(
            "\nUpdated plan:"
        )

        for index, task in enumerate(
            state.plan
        ):

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}] "
                f"(retries: "
                f"{task.get('retries', 0)})"
            )

        # --------------------------------------------------
        # Check whether everything is complete
        # --------------------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            # ----------------------------------------------
            # Generate final answer
            # ----------------------------------------------

            state.final_answer = (
                build_final_answer(state)
            )

            # ----------------------------------------------
            # Save final state
            # ----------------------------------------------

            save_checkpoint(
                state,
                run_id
            )

            return state.final_answer

    # --------------------------------------------------
    # Agent could not finish
    # --------------------------------------------------

    print(
        "\nAgent could not complete "
        "all tasks."
    )

    # Save the unfinished state so it can
    # potentially be resumed later.

    save_checkpoint(
        state,
        run_id
    )

    return (
        "I could not complete the research."
    )