import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


client = Groq(api_key=GROQ_API_KEY)

MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 8000
MAX_MESSAGES = 8


def prepare_tool_result(result):
    """
    Convert tool result to JSON and limit its size
    before adding it to the LLM context.
    """

    content = json.dumps(
        result,
        ensure_ascii=False
    )

    if len(content) > MAX_TOOL_RESULT_CHARS:

        print(
            f"Tool result too large "
            f"({len(content)} chars). "
            f"Truncating to {MAX_TOOL_RESULT_CHARS}."
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

    This prevents the LLM context from growing
    indefinitely.
    """

    if len(messages) <= MAX_MESSAGES:
        return messages

    system_message = messages[0]
    user_message = messages[1]

    recent_messages = messages[-(MAX_MESSAGES - 2):]

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

    messages = [

        {
            "role": "system",
            "content": """
You are an autonomous research worker.

Your job is to investigate the assigned research
question using the available tools.

Available tools:

1. web_search
Arguments:
{"query": "search query"}

2. fetch_page
Arguments:
{"url": "webpage URL"}

3. calculator
Arguments:
{"expression": "mathematical expression"}

4. get_weather
Arguments:
{"city": "city name"}

IMPORTANT TOOL ARGUMENT RULES:

- web_search MUST use {"query": "..."}
- fetch_page MUST use {"url": "..."}
- calculator MUST use {"expression": "..."}
- get_weather MUST use {"city": "..."}
- Never use "id", "results", "search_results",
  "input", or other incorrect argument names.

RESEARCH PROCESS:

1. Understand the research question.
2. Check previous task results.
3. Use previous results when relevant.
4. Search when information is missing.
5. Fetch useful pages when necessary.
6. Evaluate the evidence.
7. Search again only when necessary.
8. Stop when enough evidence has been collected.

Do NOT continue searching simply because
another step is available.

The maximum number of steps is a safety limit,
not a target.

When enough information is available, stop using
tools and provide the answer.

Do not invent facts.
"""
        },

        {
            "role": "user",
            "content": (
                f"Research task:\n{question}"
                f"{dependency_context}"
            )
        }
    ]

    used_search_queries = set()
    used_urls = set()

    for step in range(1, MAX_STEPS + 1):

        print(
            f"\nResearch worker step {step}"
        )

        # ----------------------------------------------
        # Prevent context from growing indefinitely
        # ----------------------------------------------

        messages = trim_messages(messages)

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800,
            )

        except Exception as error:

            print(
                f"Research worker LLM error: {error}"
            )

            return {
                "status": "failed",
                "error": str(error)
            }

        message = response.choices[0].message

        # ----------------------------------------------
        # Worker has enough information
        # ----------------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content or ""
            }

        # ----------------------------------------------
        # Process tool calls
        # ----------------------------------------------

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                print(
                    f"Invalid JSON arguments for {name}"
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or ""
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The arguments for {name} "
                            "were invalid JSON. "
                            "Retry using the exact "
                            "tool schema."
                        )
                    }
                )

                continue

            print(
                f"Tool call: {name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # ------------------------------------------
            # Prevent duplicate searches
            # ------------------------------------------

            if name == "web_search":

                query = arguments.get("query")

                if not query:

                    print(
                        "web_search called without query"
                    )

                    continue

                normalized_query = (
                    query.strip().lower()
                )

                if normalized_query in used_search_queries:

                    print(
                        "Skipping duplicate search"
                    )

                    continue

                used_search_queries.add(
                    normalized_query
                )

            # ------------------------------------------
            # Prevent duplicate page fetches
            # ------------------------------------------

            if name == "fetch_page":

                url = arguments.get("url")

                if not url:

                    print(
                        "fetch_page called without URL"
                    )

                    continue

                if url in used_urls:

                    print(
                        "Skipping duplicate URL"
                    )

                    continue

                used_urls.add(url)

            # ------------------------------------------
            # Execute tool
            # ------------------------------------------

            try:

                result = await execute_tool_async(
                    name,
                    arguments
                )

            except Exception as error:

                print(
                    f"Tool execution error: {error}"
                )

                result = {
                    "error": (
                        f"Tool execution failed: "
                        f"{error}"
                    )
                }

            # ------------------------------------------
            # Summarize webpage
            # ------------------------------------------

            if (
                name == "fetch_page"
                and isinstance(result, dict)
                and "content" in result
            ):

                print(
                    "Summarizing fetched webpage..."
                )

                try:

                    summary = summarize_content(
                        content=result["content"],
                        question=question
                    )

                    result = {
                        "url": result.get("url"),
                        "summary": summary
                    }

                except Exception as error:

                    result = {
                        "url": result.get("url"),
                        "error": (
                            "Could not summarize "
                            f"webpage: {error}"
                        )
                    }

            # ------------------------------------------
            # Add assistant tool call
            # ------------------------------------------

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": (
                                    tool_call.function.arguments
                                )
                            }
                        }
                    ]
                }
            )

            # ------------------------------------------
            # Add tool result
            # ------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": prepare_tool_result(
                        result
                    )
                }
            )

    # ----------------------------------------------
    # Maximum steps reached
    # ----------------------------------------------

    print(
        "Research worker reached maximum steps."
    )

    return {
        "status": "max_steps",
        "answer": (
            "Research could not be completed "
            "within the maximum number of steps."
        )
    }