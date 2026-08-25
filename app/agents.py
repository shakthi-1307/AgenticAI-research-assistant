import json
import time
from dataclasses import dataclass, field

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool


client = Groq(api_key=GROQ_API_KEY)

MAX_ITERATIONS = 5
MAX_TOOL_RETRIES = 2
MAX_TOOL_RESULT_CHARS = 12000


@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    iteration: int = 0
    final_answer: str | None = None
    plan: list = field(default_factory=list)


def call_llm(messages):

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1500,
    )

    return response.choices[0].message


def create_plan(question: str):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Break the user's request into the smallest
number of concrete tasks required to answer it.

Return ONLY valid JSON.

Format:

{
    "tasks": [
        "task 1",
        "task 2"
    ]
}
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format={
            "type": "json_object"
        },
        max_tokens=500,
    )

    data = json.loads(
        response.choices[0].message.content
    )

    return [
        {
            "task": task,
            "status": "pending",
        }
        for task in data["tasks"]
    ]


def limit_tool_result(result):

    if not isinstance(result, str):
        result = str(result)

    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result

    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + "\n\n"
        "[Tool result truncated because it was too large.]"
    )


def execute_tool_with_retry(name, arguments):

    for attempt in range(MAX_TOOL_RETRIES + 1):

        try:

            print(
                f"Executing {name} "
                f"(attempt {attempt + 1})"
            )

            result = execute_tool(
                name,
                arguments,
            )

            return limit_tool_result(result)

        except Exception as error:

            print(
                f"Tool '{name}' failed: {error}"
            )

            if attempt == MAX_TOOL_RETRIES:

                return {
                    "error": (
                        f"Tool '{name}' failed after "
                        f"{MAX_TOOL_RETRIES + 1} attempts."
                    )
                }

            time.sleep(1)


def execute_tools(message, state):

    results = []

    for tool_call in message.tool_calls:

        name = tool_call.function.name

        try:

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"Tool: {name}")
            print(f"Arguments: {arguments}")

            result = execute_tool_with_retry(
                name,
                arguments
            )

        except Exception as error:

            print(
                f"Tool preparation failed: {error}"
            )

            result = {
                "error": str(error)
            }

        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result),
            }
        )

        results.append(
            {
                "tool": name,
                "arguments": arguments,
                "result": result,
            }
        )

    return results


def update_plan(state, tool_results):

    """
    Update only the tasks that have direct evidence
    from the tools that were actually executed.
    """

    for item in state.plan:

        task = item["task"].lower()

        for tool_result in tool_results:

            tool_name = tool_result["tool"]
            result = tool_result["result"]

            # Never mark a task complete if the tool failed.
            if (
                isinstance(result, dict)
                and "error" in result
            ):
                continue

            # Weather task.
            if (
                tool_name == "get_weather"
                and "weather" in task
            ):
                item["status"] = "complete"

            # Calculation task.
            elif (
                tool_name == "calculator"
                and (
                    "calcul" in task
                    or "multiply" in task
                    or "product" in task
                )
            ):
                item["status"] = "complete"


def all_tasks_complete(state):

    if not state.plan:
        return False

    return all(
        item["status"] == "complete"
        for item in state.plan
    )


def build_final_answer(state):

    """
    Ask the LLM to produce the final response
    using the completed research results.
    """

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


def research(question: str) -> str:

    state = AgentState(
        messages=[
            {
                "role": "system",
                "content": """
You are an autonomous research assistant.

You have a plan describing the tasks required
to answer the user's question.

For every step:

1. Look at the current plan.
2. Choose the appropriate tool.
3. Do not repeat completed work.
4. Use calculator for mathematical calculations.
5. Use get_weather for weather questions.
6. Use web_search for general web research.
7. Use fetch_page when more detail is needed.
8. Do not invent facts or sources.
9. Stop when all tasks are complete.
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    # -----------------------------------------
    # 1. Create plan
    # -----------------------------------------

    state.plan = create_plan(question)

    print("\nPlan:")

    for index, item in enumerate(
        state.plan,
        start=1
    ):
        print(
            f"{index}. {item['task']} "
            f"[{item['status']}]"
        )

    # Give the plan to the LLM.
    state.messages.append(
        {
            "role": "system",
            "content": (
                "Current plan:\n"
                + json.dumps(
                    state.plan,
                    indent=2
                )
            ),
        }
    )

    # -----------------------------------------
    # 2. Agent loop
    # -----------------------------------------

    while state.iteration < MAX_ITERATIONS:

        state.iteration += 1

        print(
            f"\n--- Agent iteration "
            f"{state.iteration} ---"
        )

        message = call_llm(
            state.messages
        )

        state.messages.append(message)

        # LLM decided it can answer.
        if not message.tool_calls:

            state.final_answer = message.content

            break

        # Execute tools.
        tool_results = execute_tools(
            message,
            state
        )

        # Update plan using actual tool evidence.
        update_plan(
            state,
            tool_results
        )

        print("\nUpdated plan:")

        for index, item in enumerate(
            state.plan,
            start=1
        ):
            print(
                f"{index}. {item['task']} "
                f"[{item['status']}]"
            )

        # Deterministic completion check.
        if all_tasks_complete(state):

            print(
                "\nAll tasks completed."
            )

            state.final_answer = (
                build_final_answer(state)
            )

            break

    if state.final_answer:

        return state.final_answer

    return (
        "I could not complete the research within "
        "the maximum number of research steps."
    )