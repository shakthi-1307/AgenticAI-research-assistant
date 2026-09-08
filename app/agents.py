import json
from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .context import build_context
from .executor import execute_ready_tasks
from .planner import create_plan, get_ready_tasks, all_tasks_complete
from .checkpoint import (
    create_run_id,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint
)
from .state import AgentState


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5


def save_result(state, result):
    state.results.append({
        "task": result["task"]["task"],
        "type": result["task"]["type"],
        "result": result["result"]
    })


def get_working_memory(state):
    return state.results


def task_succeeded(task, result):
    """
    Decide whether a worker actually completed its task.
    """

    if not isinstance(result, dict):
        return False

    # Explicit worker failure
    if result.get("status") == "failed":
        return False

    # Explicit error
    if "error" in result:
        return False

    # Research workers must explicitly say complete
    if task["type"] == "research":
        return result.get("status") == "complete"

    # Calculation workers must explicitly say complete
    if task["type"] == "calculation":
        return result.get("status") == "complete"

    # Weather workers
    if task["type"] == "weather":
        return result.get("status") == "complete"

    return False


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
only the information collected by the agent.

Rules:
- Use only provided context.
- Do not invent information.
- Identify estimates.
- Be concise.
- Do not use tools.
"""
            },
            {
                "role": "user",
                "content": (
                    f"Original question:\n{state.question}\n\n"
                    f"Agent context:\n\n{context}"
                )
            }
        ],
        tool_choice="none",
        max_tokens=1500
    )

    return response.choices[0].message.content


async def research(question, run_id=None):

    if run_id is None:
        run_id = create_run_id()

    print(f"\nRun ID: {run_id}")

    checkpoint_state = load_checkpoint(run_id)

    # ---------------------------------------------------------
    # Resume existing run
    # ---------------------------------------------------------

    if (
        checkpoint_state is not None
        and checkpoint_state.question.strip() == question.strip()
    ):
        state = checkpoint_state

        print("\nResuming from checkpoint...")
        print(f"Previous iteration: {state.iteration}")

        print("\nRestored plan:")

        for index, task in enumerate(state.plan):
            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

    # ---------------------------------------------------------
    # Start new run
    # ---------------------------------------------------------

    else:

        if checkpoint_state is not None:

            print("\nCheckpoint belongs to a different question.")
            print("Starting a new state.")

            delete_checkpoint(run_id)

        state = AgentState(
            question=question,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        state.plan = create_plan(question)

        print("\nPlan:")

        for index, task in enumerate(state.plan):

            dependencies = task.get("depends_on", [])

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}]"
            )

            if dependencies:
                print(
                    f"   depends_on: {dependencies}"
                )

    # ---------------------------------------------------------
    # Agent execution loop
    # ---------------------------------------------------------

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        print(
            f"\n--- Agent iteration "
            f"{state.iteration} ---"
        )

        ready_tasks = get_ready_tasks(state.plan)

        print(
            f"Ready tasks: {len(ready_tasks)}"
        )

        if not ready_tasks:
            break

        # Execute independent tasks in parallel
        results = await execute_ready_tasks(
            ready_tasks,
            state.plan
        )

        # -----------------------------------------------------
        # Process worker results
        # -----------------------------------------------------

        for item in results:

            task = item["task"]
            result = item["result"]

            success = task_succeeded(
                task,
                result
            )

            if success:

                task["status"] = "complete"
                task["result"] = result

                print(
                    f"Task completed: "
                    f"{task['task']}"
                )

            else:

                task["status"] = "failed"

                task["retries"] = (
                    task.get("retries", 0) + 1
                )

                print(
                    f"Task failed: "
                    f"{task['task']} "
                    f"(retry {task['retries']})"
                )

                if isinstance(result, dict):
                    print(
                        f"   Error: "
                        f"{result.get('error', result)}"
                    )

            save_result(
                state,
                item
            )

        # -----------------------------------------------------
        # Save checkpoint
        # -----------------------------------------------------

        save_checkpoint(
            state,
            run_id
        )

        # -----------------------------------------------------
        # Working memory
        # -----------------------------------------------------

        print("\nWorking memory:")

        print(
            json.dumps(
                get_working_memory(state),
                indent=2,
                ensure_ascii=False
            )
        )

        # -----------------------------------------------------
        # Updated plan
        # -----------------------------------------------------

        print("\nUpdated plan:")

        for index, task in enumerate(state.plan):

            print(
                f"{index + 1}. "
                f"{task['task']} "
                f"[{task['status']}] "
                f"(retries: "
                f"{task.get('retries', 0)})"
            )

        # -----------------------------------------------------
        # Everything completed
        # -----------------------------------------------------

        if all_tasks_complete(state.plan):

            print("\nAll tasks completed.")

            state.final_answer = build_final_answer(
                state
            )

            save_checkpoint(
                state,
                run_id
            )

            return state.final_answer

    # ---------------------------------------------------------
    # Could not finish
    # ---------------------------------------------------------

    print(
        "\nAgent could not complete all tasks."
    )

    save_checkpoint(
        state,
        run_id
    )

    return "I could not complete the research."