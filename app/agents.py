import json
from dataclasses import dataclass, field

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .context import build_context
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


# ============================================================
# AGENT STATE
# ============================================================

@dataclass
class AgentState:

    # Conversation/context visible to the parent LLM
    messages: list = field(
        default_factory=list
    )

    # Tasks created by the planner
    plan: list = field(
        default_factory=list
    )

    # Results produced by workers
    results: list = field(
        default_factory=list
    )

    # Runtime information
    iteration: int = 0

    # Final answer
    final_answer: str | None = None


# ============================================================
# SAVE WORKER RESULT
# ============================================================

def save_result(state, result):
    """
    Store the result returned by a worker
    in the parent agent's working memory.
    """

    state.results.append(
        {
            "task": result["task"]["task"],
            "type": result["task"]["type"],
            "result": result["result"],
        }
    )


# ============================================================
# WORKING MEMORY
# ============================================================

def get_working_memory(state):
    """
    Return the results collected
    during the current agent run.
    """

    return state.results


# ============================================================
# FINAL ANSWER
# ============================================================

def build_final_answer(state):
    """
    Ask the parent LLM to synthesize
    the worker results into the final answer.
    """

    context = build_context(state)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are the final answer generator.

Answer the user's original question using
the information collected by the agents.

Rules:

- Use only the provided agent context.
- Do not call tools.
- Do not invent information.
- If a task failed, clearly acknowledge it.
- Give a concise and useful answer.
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


# ============================================================
# PARENT AGENT
# ============================================================

async def research(question: str):

    """
    Main parent agent.

    Responsibilities:

    1. Create the plan.
    2. Find tasks that are ready.
    3. Send ready tasks to the executor.
    4. Receive worker results.
    5. Update task states.
    6. Store results in working memory.
    7. Generate the final answer.
    """

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    state = AgentState(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    # --------------------------------------------------------
    # 1. CREATE PLAN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. AGENT EXECUTION LOOP
    # --------------------------------------------------------

    while (
        state.iteration
        < MAX_ITERATIONS
    ):

        state.iteration += 1

        # ----------------------------------------------------
        # FIND TASKS THAT CAN RUN
        # ----------------------------------------------------

        ready_tasks = get_ready_tasks(
            state.plan
        )

        print(
            f"\nReady tasks: "
            f"{len(ready_tasks)}"
        )

        # No tasks available
        if not ready_tasks:
            break

        # ----------------------------------------------------
        # EXECUTE READY TASKS
        # ----------------------------------------------------

        results = await execute_ready_tasks(
            ready_tasks
        )

        # ----------------------------------------------------
        # PROCESS WORKER RESULTS
        # ----------------------------------------------------

        for item in results:

            task = item["task"]
            result = item["result"]

            # -----------------------------------------------
            # RESEARCH TASK
            # -----------------------------------------------

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

                    task["status"] = (
                        "failed"
                    )

            # -----------------------------------------------
            # OTHER TASKS
            # -----------------------------------------------

            else:

                if (
                    isinstance(result, dict)
                    and "error" not in result
                ):

                    task["status"] = (
                        "complete"
                    )

                else:

                    task["status"] = (
                        "failed"
                    )

            # -----------------------------------------------
            # SAVE RESULT
            # -----------------------------------------------

            save_result(
                state,
                item
            )

        # ----------------------------------------------------
        # SHOW WORKING MEMORY
        # ----------------------------------------------------

        print(
            "\nWorking memory:"
        )

        print(
            json.dumps(
                state.results,
                indent=2
            )
        )

        # ----------------------------------------------------
        # SHOW UPDATED PLAN
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CHECK COMPLETION
        # ----------------------------------------------------

        if all_tasks_complete(
            state.plan
        ):

            print(
                "\nAll tasks completed."
            )

            break

    # --------------------------------------------------------
    # 3. GENERATE FINAL ANSWER
    # --------------------------------------------------------

    if all_tasks_complete(
        state.plan
    ):

        state.final_answer = (
            build_final_answer(
                state
            )
        )

        return state.final_answer

    # --------------------------------------------------------
    # FAILED
    # --------------------------------------------------------

    return (
        "I could not complete the research."
    )