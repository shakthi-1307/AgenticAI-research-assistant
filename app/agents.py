import json
import os

from .llm import call_llm
from .planner import create_plan, get_ready_tasks, all_tasks_complete
from .executor import execute_ready_tasks
from .state import AgentState
from .checkpoint import (
    create_run_id,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
)
from .context import build_context


MAX_ITERATIONS = 5


def save_result(state, task, result):
    """
    Save a completed task result into working memory.
    """

    state.results.append({
        "task": task["task"],
        "type": task["type"],
        "result": result
    })


def get_working_memory(state):
    """
    Return the accumulated task results.
    """

    return state.results


def task_succeeded(task, result):
    """
    Determine whether a worker completed successfully.
    """

    if not isinstance(result, dict):
        return False

    if result.get("status") != "complete":
        return False

    if result.get("error"):
        return False

    return True


def build_final_answer(state):
    """
    Synthesize the final answer from completed task results.
    """

    context = build_context(state)

    messages = [
        {
            "role": "system",
            "content": """
You are the final answer synthesizer for an AI research assistant.

Use only the information provided in the research context.

Rules:
- Directly answer the user's original question.
- Combine information from the completed tasks.
- Do not invent facts.
- Do not mention internal agents, workers, checkpoints,
  task graphs, or implementation details unless relevant
  to the user's question.
- If information is incomplete, clearly say so.
- Keep the answer concise but informative.
- Use clear structure when useful.
"""
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{state.question}\n\n"
                f"Research context:\n{context}"
            )
        }
    ]

    message = call_llm(
        messages=messages,
        tools=None,
        tool_choice="none",
        max_tokens=1500
    )

    return message.content or ""


async def research(question, run_id=None):
    """
    Main orchestration loop.

    Responsibilities:
    - Create or restore state
    - Generate the plan
    - Execute ready tasks
    - Update task state
    - Handle retries
    - Save checkpoints
    - Produce the final answer
    """

    # ---------------------------------------------------------
    # Create or restore run
    # ---------------------------------------------------------

    if run_id is None:
        run_id = create_run_id()

    checkpoint = load_checkpoint(run_id)

    if checkpoint:
        state = checkpoint

        # Prevent accidentally resuming a different question.
        if state.question != question:
            print("Checkpoint question does not match current question.")
            delete_checkpoint(run_id)

            state = AgentState(
                question=question
            )

    else:
        state = AgentState(
            question=question
        )

    print(f"\nRun ID: {run_id}")

    # ---------------------------------------------------------
    # Create plan if needed
    # ---------------------------------------------------------

    if not state.plan:
        print("\nCreating plan...")

        state.plan = create_plan(question)

        save_checkpoint(state, run_id)

    # ---------------------------------------------------------
    # Main orchestration loop
    # ---------------------------------------------------------

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        print(
            f"\n=============================="
            f"\nIteration {state.iteration}"
            f"\n=============================="
        )

        # -----------------------------------------------------
        # Check whether everything is already complete
        # -----------------------------------------------------

        if all_tasks_complete(state.plan):

            print("\nAll tasks completed.")

            state.final_answer = build_final_answer(state)

            save_checkpoint(state, run_id)

            return state.final_answer

        # -----------------------------------------------------
        # Find tasks whose dependencies are satisfied
        # -----------------------------------------------------

        ready_tasks = get_ready_tasks(state.plan)

        if not ready_tasks:

            print("\nNo tasks are currently ready.")

            save_checkpoint(state, run_id)

            return (
                "Research could not continue because no executable "
                "tasks were available."
            )

        print(f"\nReady tasks: {len(ready_tasks)}")

        # -----------------------------------------------------
        # Execute ready tasks
        # -----------------------------------------------------

        execution_results = await execute_ready_tasks(
            ready_tasks,
            state.plan
        )

        # -----------------------------------------------------
        # Commit execution results into AgentState
        # -----------------------------------------------------

        for execution in execution_results:

            task = execution["task"]
            result = execution["result"]

            task_index = state.plan.index(task)

            if task_succeeded(task, result):

                task["status"] = "complete"
                task["result"] = result

                save_result(
                    state,
                    task,
                    result
                )

                print(
                    f"Task completed: {task['task']}"
                )

            else:

                task["status"] = "failed"
                task["result"] = result

                task["retries"] = task.get("retries", 0) + 1

                print(
                    f"Task failed: {task['task']}"
                )

                print(
                    f"Retry {task['retries']}/"
                    f"{task.get('max_retries', 2)}"
                )

                # If retry limit has not been reached,
                # get_ready_tasks() will allow it to run again.
                if task["retries"] < task.get("max_retries", 2):
                    task["status"] = "pending"

        # -----------------------------------------------------
        # Save checkpoint after this execution batch
        # -----------------------------------------------------

        save_checkpoint(
            state,
            run_id
        )

        # -----------------------------------------------------
        # Display working memory
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
        # Display current plan
        # -----------------------------------------------------

        print("\nCurrent plan:")

        print(
            json.dumps(
                state.plan,
                indent=2,
                ensure_ascii=False
            )
        )

    # ---------------------------------------------------------
    # Maximum iterations reached
    # ---------------------------------------------------------

    save_checkpoint(
        state,
        run_id
    )

    return (
        "Research could not be completed within the maximum "
        "number of iterations."
    )