import json
from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


client = Groq(api_key=GROQ_API_KEY)

MAX_STEPS = 5
MAX_TOOL_RESULT_CHARS = 12000


def prepare_tool_result(result):
    """
    Convert a tool result into a bounded string
    before sending it back to the LLM.
    """

    content = json.dumps(
        result,
        ensure_ascii=False,
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


async def research_task(task):
    """
    Research worker agent.

    The worker:
    1. Receives a research task.
    2. Decides which tools to use.
    3. Executes tools.
    4. Summarizes fetched webpages.
    5. Uses the collected evidence to produce a final result.
    """

    question = task["task"]

    messages = [
        {
            "role": "system",
            "content": """
You are a research worker agent.

Your job is to research the user's question using the
available tools and produce a concise evidence-based answer.

Available tools:

1. web_search

Arguments MUST be exactly:

{
    "query": "your search query"
}

IMPORTANT:
- The argument name is ALWAYS "query".
- NEVER use "id".
- NEVER use "search_results".
- NEVER use "results".
- NEVER use "search".
- NEVER call web_search without a query.

Example:

{
    "query": "what is artificial intelligence"
}

2. fetch_page

Arguments MUST be exactly:

{
    "url": "https://example.com"
}

3. calculator

Arguments MUST be exactly:

{
    "expression": "2 + 3"
}

4. get_weather

Arguments MUST be exactly:

{
    "city": "Chennai"
}

Research process:

1. Start with web_search when external information is needed.
2. Examine the search results.
3. If a result contains a useful webpage, use fetch_page.
4. Fetched webpages will be summarized separately.
5. Use the summaries as evidence.
6. Avoid repeatedly searching for the exact same thing.
7. Do not call unnecessary tools.
8. Once you have enough information, stop using tools and answer.

Do not invent information.

If the available evidence is insufficient, say so.

Keep the final answer concise and factual.
""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    used_search_queries = set()
    used_urls = set()

    for step in range(1, MAX_STEPS + 1):

        print(f"Research worker step {step}")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1200,
            )

        except Exception as error:
            print(f"Research worker LLM error: {error}")

            return {
                "status": "failed",
                "error": f"Research worker LLM failed: {error}",
            }

        message = response.choices[0].message

        # --------------------------------------------------
        # No tool call -> worker has finished researching
        # --------------------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content or "",
            }

        # --------------------------------------------------
        # Process tool calls
        # --------------------------------------------------

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError as error:

                print(
                    f"Invalid tool arguments for {name}: "
                    f"{error}"
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous tool call for "
                            f"{name} had invalid JSON arguments. "
                            f"Please retry with valid JSON."
                        ),
                    }
                )

                continue

            print(
                f"Research worker tool call: "
                f"{name} {arguments}"
            )

            # --------------------------------------------------
            # Prevent duplicate web searches
            # --------------------------------------------------

            if name == "web_search":

                query = arguments.get("query")

                if not query:
                    print(
                        "Invalid web_search call: "
                        "missing query."
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content or "",
                        }
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The web_search tool requires "
                                'arguments in this exact form: '
                                '{"query": "your search query"}. '
                                "Please provide a valid query."
                            ),
                        }
                    )

                    continue

                normalized_query = query.strip().lower()

                if normalized_query in used_search_queries:

                    print(
                        f"Skipping duplicate search: {query}"
                    )

                    continue

                used_search_queries.add(normalized_query)

            # --------------------------------------------------
            # Prevent duplicate webpage fetching
            # --------------------------------------------------

            if name == "fetch_page":

                url = arguments.get("url")

                if not url:
                    print(
                        "Invalid fetch_page call: "
                        "missing url."
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content or "",
                        }
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The fetch_page tool requires "
                                'arguments in this exact form: '
                                '{"url": "https://example.com"}. '
                                "Please provide a valid URL."
                            ),
                        }
                    )

                    continue

                if url in used_urls:

                    print(
                        f"Skipping duplicate URL: {url}"
                    )

                    continue

                used_urls.add(url)

            # --------------------------------------------------
            # Execute the tool
            # --------------------------------------------------

            try:
                result = await execute_tool_async(
                    name,
                    arguments,
                )

            except Exception as error:

                result = {
                    "error": (
                        f"Tool execution failed: {error}"
                    )
                }

            # --------------------------------------------------
            # Fetch page -> summarize page
            # --------------------------------------------------

            if (
                name == "fetch_page"
                and isinstance(result, dict)
                and "content" in result
            ):

                print("Summarizing fetched webpage...")

                try:

                    summary = summarize_content(
                        content=result["content"],
                        question=question,
                    )

                    result = {
                        "url": result.get("url"),
                        "summary": summary,
                    }

                except Exception as error:

                    print(
                        f"Summarization failed: {error}"
                    )

                    result = {
                        "url": result.get("url"),
                        "error": (
                            f"Could not summarize webpage: "
                            f"{error}"
                        ),
                    }

            # --------------------------------------------------
            # Add assistant's tool call to conversation
            # --------------------------------------------------

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
                                ),
                            },
                        }
                    ],
                }
            )

            # --------------------------------------------------
            # Add bounded tool result
            # --------------------------------------------------

            safe_result = prepare_tool_result(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": safe_result,
                }
            )

    # ------------------------------------------------------
    # Maximum research steps reached
    # ------------------------------------------------------

    print(
        "Research worker reached maximum steps."
    )

    return {
        "status": "complete",
        "answer": (
            "Research completed, but the worker reached "
            "its maximum number of research steps."
        ),
    }