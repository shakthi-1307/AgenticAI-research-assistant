import json

from .llm import call_llm
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


MAX_STEPS = 6

# Maximum size of an individual tool result
MAX_TOOL_RESULT_CHARS = 6000

# Maximum size of the complete LLM conversation
MAX_CONTEXT_CHARS = 14000

# Maximum number of previous messages to retain
MAX_MESSAGES = 8


# =========================================================
# Tool Result Preparation
# =========================================================

def prepare_tool_result(result):
    """
    Convert a tool result into a compact string.

    Prevents large tool outputs from consuming the
    entire LLM context.
    """

    result_text = json.dumps(
        result,
        ensure_ascii=False
    )

    if len(result_text) > MAX_TOOL_RESULT_CHARS:

        result_text = (
            result_text[:MAX_TOOL_RESULT_CHARS]
            + "\n...[tool result truncated]"
        )

    return result_text


# =========================================================
# Dependency Context
# =========================================================

def build_dependency_context(dependency_results):
    """
    Build compact context from completed dependency tasks.
    """

    if not dependency_results:
        return ""

    context = json.dumps(
        dependency_results,
        indent=2,
        ensure_ascii=False
    )

    # Prevent dependency results from becoming huge.
    if len(context) > 5000:
        context = (
            context[:5000]
            + "\n...[dependency context truncated]"
        )

    return (
        "\n\nInformation from completed tasks:\n"
        + context
    )


# =========================================================
# Message Trimming
# =========================================================

def trim_messages(messages):
    """
    Keep the system prompt and user request while
    removing excessive historical context.
    """

    if len(messages) <= MAX_MESSAGES:

        messages = messages.copy()

    else:

        messages = (
            [messages[0]]
            + messages[-(MAX_MESSAGES - 1):]
        )

    # -----------------------------------------------------
    # Enforce character budget
    # -----------------------------------------------------

    total_chars = sum(
        len(str(message.get("content", "")))
        for message in messages
    )

    if total_chars <= MAX_CONTEXT_CHARS:

        return messages

    # Always preserve system message.
    system_message = messages[0]

    remaining = messages[1:]

    trimmed = [system_message]

    current_chars = len(
        str(system_message.get("content", ""))
    )

    # Keep the newest messages first.
    for message in reversed(remaining):

        content = str(
            message.get("content", "")
        )

        if (
            current_chars + len(content)
            > MAX_CONTEXT_CHARS
        ):

            continue

        trimmed.insert(1, message)

        current_chars += len(content)

    return trimmed


# =========================================================
# Research Worker
# =========================================================

async def research_task(
    task,
    dependency_results=None
):
    """
    Autonomous research worker.

    Loop:

        Observe
          ↓
        Decide
          ↓
        Act
          ↓
        Observe
          ↓
        Decide
          ↓
        ...

    until the research is complete or MAX_STEPS is reached.
    """

    dependency_results = dependency_results or []

    question = task["task"]

    dependency_context = build_dependency_context(
        dependency_results
    )

    # -----------------------------------------------------
    # Initial conversation
    # -----------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": """
You are an autonomous research agent.

Your job is to research the user's question using
the available tools and produce an evidence-based answer.

Available tools:
- web_search(query): search the web for relevant information.
- fetch_page(url): retrieve webpage content.
- calculator(expression): perform arithmetic.
- get_weather(city): retrieve weather information.

Research strategy:
1. Start with web_search.
2. Inspect search results.
3. Fetch useful pages when necessary.
4. Use additional searches when information is incomplete
   or needs verification.
5. Stop researching when you have enough reliable evidence.
6. Then provide a concise final answer.

Important:
- Do not invent information.
- Prefer reliable and authoritative sources.
- Verify important claims when possible.
- Do not repeatedly search the same query.
- Do not repeatedly fetch the same URL.
- When you have enough evidence, answer directly.
- If evidence is insufficient, clearly state that.

Tool arguments must exactly match their schemas.
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

    # -----------------------------------------------------
    # Duplicate prevention
    # -----------------------------------------------------

    used_search_queries = set()

    used_urls = set()

    # =====================================================
    # Autonomous research loop
    # =====================================================

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        print(
            f"\nResearch worker step {step}"
        )

        # -------------------------------------------------
        # Control context size before every LLM call
        # -------------------------------------------------

        messages = trim_messages(
            messages
        )

        try:

            message = call_llm(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800,
                component="research_worker"
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

        # -------------------------------------------------
        # No tool call means the model has finished
        # -------------------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content or ""
            }

        # -------------------------------------------------
        # Process tool calls
        # -------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as error:

                print(
                    f"Invalid tool arguments: "
                    f"{error}"
                )

                continue

            print(
                f"Tool call: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # =================================================
            # Duplicate search prevention
            # =================================================

            if tool_name == "web_search":

                query = arguments.get(
                    "query",
                    ""
                ).strip()

                normalized_query = query.lower()

                if (
                    normalized_query
                    in used_search_queries
                ):

                    print(
                        "Skipping duplicate search."
                    )

                    continue

                used_search_queries.add(
                    normalized_query
                )

            # =================================================
            # Duplicate URL prevention
            # =================================================

            if tool_name == "fetch_page":

                url = arguments.get(
                    "url",
                    ""
                ).strip()

                if url in used_urls:

                    print(
                        "Skipping duplicate URL."
                    )

                    continue

                used_urls.add(url)

            # =================================================
            # Execute tool
            # =================================================

            try:

                result = await execute_tool_async(
                    tool_name,
                    arguments
                )

            except Exception as error:

                result = {
                    "status": "failed",
                    "error": str(error)
                }

            # -------------------------------------------------
            # Summarize webpage content
            # -------------------------------------------------

            if (
                tool_name == "fetch_page"
                and isinstance(result, dict)
                and result.get("content")
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

                    print(
                        f"Summarization error: "
                        f"{error}"
                    )

                    result = {
                        "url": result.get("url"),
                        "content": result["content"][
                            :MAX_TOOL_RESULT_CHARS
                        ]
                    }

            # -------------------------------------------------
            # Prepare compact tool result
            # -------------------------------------------------

            result_text = prepare_tool_result(
                result
            )

            # -------------------------------------------------
            # Add assistant tool call
            # -------------------------------------------------

            messages.append(
                {
                    "role": "assistant",
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
            )

            # -------------------------------------------------
            # Add tool result
            # -------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                }
            )

    # =====================================================
    # Maximum research steps reached
    # =====================================================

    return {
        "status": "max_steps",
        "answer": (
            "Research could not be completed within "
            "the maximum number of research steps."
        )
    }