import json

from .llm import call_llm
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 8000
MAX_MESSAGES = 8


def prepare_tool_result(result):
    """
    Convert a tool result into a string and
    prevent excessively large results from
    entering the LLM context.
    """

    content = json.dumps(
        result,
        ensure_ascii=False
    )

    if len(content) > MAX_TOOL_RESULT_CHARS:

        print(
            "Tool result too large. "
            "Truncating before sending to LLM."
        )

        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + "\n...[tool result truncated]"
        )

    return content


def build_dependency_context(dependency_results):

    if not dependency_results:
        return ""

    return (
        "\n\nResults from previous tasks:\n"
        + json.dumps(
            dependency_results,
            ensure_ascii=False,
            indent=2
        )
    )


def trim_messages(messages):
    """
    Keep the system prompt, original user task,
    and only the most recent conversation messages.
    """

    if len(messages) <= MAX_MESSAGES:
        return messages

    system_message = messages[0]
    user_message = messages[1]

    recent_messages = messages[
        -(MAX_MESSAGES - 2):
    ]

    return [
        system_message,
        user_message,
        *recent_messages
    ]


async def research_task(
    task,
    dependency_results=None
):

    dependency_results = dependency_results or []

    question = task["task"]

    dependency_context = build_dependency_context(
        dependency_results
    )

    # ---------------------------------------------------------
    # Initial research context
    # ---------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": """
You are an autonomous research worker
inside an Agentic AI system.

Your job is to research the assigned task
using the available tools.

Available tools include:

- web_search
- fetch_page
- calculator
- get_weather

Research process:

1. Understand the research task.
2. Decide what information is needed.
3. Use web_search to find relevant sources.
4. Use fetch_page when a source needs to be examined.
5. Analyze the tool results.
6. Continue researching if important information
   is still missing.
7. Stop when you have enough reliable information.
8. Return a concise evidence-based answer.

IMPORTANT TOOL ARGUMENTS:

web_search:
{
  "query": "your search query"
}

fetch_page:
{
  "url": "https://example.com"
}

calculator:
{
  "expression": "10 / 2"
}

get_weather:
{
  "city": "Chennai"
}

Rules:

- Use tools when they are useful.
- Do not invent information.
- Prefer reliable sources.
- Do not repeatedly search for the exact same query.
- Do not repeatedly fetch the same URL.
- Stop researching when sufficient evidence exists.
- Return a concise final answer.
"""
        },

        {
            "role": "user",
            "content": (
                f"Research task:\n"
                f"{question}"
                f"{dependency_context}"
            )
        }
    ]

    # ---------------------------------------------------------
    # Track previously used searches / URLs
    # ---------------------------------------------------------

    used_search_queries = set()
    used_urls = set()

    # ---------------------------------------------------------
    # Agent loop
    # ---------------------------------------------------------

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        print(
            f"\nResearch worker step {step}"
        )

        messages = trim_messages(
            messages
        )

        # -----------------------------------------------------
        # LLM decides what to do
        # -----------------------------------------------------

        try:

            message = call_llm(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800
            )

        except Exception as error:

            print(
                f"Research worker LLM error: "
                f"{error}"
            )

            return {
                "status": "failed",
                "error": str(error)
            }

        # -----------------------------------------------------
        # No tool call → research is complete
        # -----------------------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content or ""
            }

        # -----------------------------------------------------
        # Process tool calls
        # -----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            print(
                f"Tool call: {tool_name}"
            )

            # -------------------------------------------------
            # Parse arguments
            # -------------------------------------------------

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                print(
                    "Invalid tool arguments returned "
                    "by the LLM."
                )

                messages.append({
                    "role": "assistant",
                    "content": (
                        "The previous tool call contained "
                        "invalid JSON arguments. "
                        "Please provide valid arguments."
                    )
                })

                continue

            print(
                f"Arguments: {arguments}"
            )

            # -------------------------------------------------
            # Duplicate search prevention
            # -------------------------------------------------

            if tool_name == "web_search":

                query = arguments.get(
                    "query",
                    ""
                ).strip().lower()

                if query in used_search_queries:

                    print(
                        "Skipping duplicate search."
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            "This search query was already "
                            "executed. Use the existing result "
                            "or try a different query."
                        )
                    })

                    continue

                used_search_queries.add(
                    query
                )

            # -------------------------------------------------
            # Duplicate URL prevention
            # -------------------------------------------------

            if tool_name == "fetch_page":

                url = arguments.get(
                    "url",
                    ""
                ).strip()

                if url in used_urls:

                    print(
                        "Skipping duplicate URL."
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            "This URL was already fetched. "
                            "Use the existing information."
                        )
                    })

                    continue

                used_urls.add(
                    url
                )

            # -------------------------------------------------
            # Execute tool
            # -------------------------------------------------

            try:

                result = await execute_tool_async(
                    tool_name,
                    arguments
                )

            except Exception as error:

                result = {
                    "error": str(error)
                }

            # -------------------------------------------------
            # Summarize webpage content
            # -------------------------------------------------

            if (
                tool_name == "fetch_page"
                and isinstance(result, dict)
                and "content" in result
            ):

                print(
                    "Summarizing fetched webpage..."
                )

                try:

                    summary = summarize_content(
                        result["content"],
                        question
                    )

                    result = {
                        "url": result.get("url"),
                        "summary": summary
                    }

                except Exception as error:

                    result = {
                        "error": (
                            "Could not summarize webpage: "
                            f"{error}"
                        )
                    }

            # -------------------------------------------------
            # Prepare tool result
            # -------------------------------------------------

            tool_content = prepare_tool_result(
                result
            )

            # -------------------------------------------------
            # Add assistant tool-call message
            # -------------------------------------------------

            assistant_message = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                arguments
                            )
                        }
                    }
                ]
            }

            messages.append(
                assistant_message
            )

            # -------------------------------------------------
            # Add tool result
            # -------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content
            })

    # ---------------------------------------------------------
    # Maximum research steps reached
    # ---------------------------------------------------------

    return {
        "status": "max_steps",
        "answer": (
            "Research could not be completed "
            "within the maximum number of steps."
        )
    }