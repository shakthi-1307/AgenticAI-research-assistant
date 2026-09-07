import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


client = Groq(api_key=GROQ_API_KEY)

# Hard safety limit.
# The agent should normally stop before reaching this.
MAX_STEPS = 6

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
    2. Decides which tools are needed.
    3. Executes tools.
    4. Summarizes fetched webpages.
    5. Evaluates whether enough evidence has been collected.
    6. Produces a final research answer.
    """

    question = task["task"]

    messages = [
        {
            "role": "system",
            "content": """
You are a research worker agent.

Your job is to research the user's question using the
available tools and produce a concise, evidence-based answer.

AVAILABLE TOOLS

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
    "query": "differences between AI and machine learning"
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


RESEARCH PROCESS

1. Start with web_search when external information is needed.

2. Examine the search results.

3. Fetch useful webpages when additional detail
   is required.

4. After every tool result, evaluate whether you
   have enough reliable evidence to answer the question.

5. If you have enough evidence, STOP using tools
   and provide the final answer.

6. Only perform another search if the existing evidence
   is insufficient, conflicting, or missing an important
   part of the question.

7. Avoid repeatedly searching for information you
   already have.

8. Do not call unnecessary tools.

9. Never continue researching simply because another
   step is available.

10. Do not invent information.

11. If the available evidence is insufficient,
    clearly say so.

12. Keep the final answer concise and factual.


STOPPING CONDITION

Your primary goal is to answer the question correctly,
not to maximize the number of searches.

If the collected evidence is sufficient:

- Do not call another tool.
- Return the final answer immediately.

The system also imposes a maximum number of research
steps as a safety limit. If that limit is reached,
stop researching rather than continuing indefinitely.
""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Keep track of searches already performed.
    used_search_queries = set()

    # Keep track of webpages already fetched.
    used_urls = set()

    for step in range(1, MAX_STEPS + 1):

        print(
            f"Research worker step {step}"
        )

        # --------------------------------------------------
        # Ask the LLM what to do next
        # --------------------------------------------------

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1200,
            )

        except Exception as error:

            print(
                f"Research worker LLM error: {error}"
            )

            return {
                "status": "failed",
                "error": (
                    f"Research worker LLM failed: "
                    f"{error}"
                ),
            }

        message = response.choices[0].message

        # --------------------------------------------------
        # No tool call means the LLM has finished
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

            # --------------------------------------------------
            # Parse tool arguments
            # --------------------------------------------------

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as error:

                print(
                    f"Invalid tool arguments for "
                    f"{name}: {error}"
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
                            f"{name} contained invalid JSON. "
                            f"Retry using valid JSON arguments."
                        ),
                    }
                )

                continue

            print(
                f"Research worker tool call: "
                f"{name} {arguments}"
            )

            # --------------------------------------------------
            # Validate web_search
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

                normalized_query = (
                    query.strip().lower()
                )

                # --------------------------------------------------
                # Prevent duplicate searches
                # --------------------------------------------------

                if normalized_query in used_search_queries:

                    print(
                        f"Skipping duplicate search: "
                        f"{query}"
                    )

                    continue

                used_search_queries.add(
                    normalized_query
                )

            # --------------------------------------------------
            # Validate fetch_page
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

                # --------------------------------------------------
                # Prevent duplicate webpage fetching
                # --------------------------------------------------

                if url in used_urls:

                    print(
                        f"Skipping duplicate URL: "
                        f"{url}"
                    )

                    continue

                used_urls.add(url)

            # --------------------------------------------------
            # Execute tool
            # --------------------------------------------------

            try:

                result = await execute_tool_async(
                    name,
                    arguments,
                )

            except Exception as error:

                result = {
                    "error": (
                        f"Tool execution failed: "
                        f"{error}"
                    )
                }

            # --------------------------------------------------
            # Fetch webpage -> summarize it
            # --------------------------------------------------

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
                        question=question,
                    )

                    result = {
                        "url": result.get("url"),
                        "summary": summary,
                    }

                except Exception as error:

                    print(
                        f"Summarization failed: "
                        f"{error}"
                    )

                    result = {
                        "url": result.get("url"),
                        "error": (
                            "Could not summarize webpage: "
                            f"{error}"
                        ),
                    }

            # --------------------------------------------------
            # Add assistant tool call to conversation
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

            safe_result = prepare_tool_result(
                result
            )

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
        "status": "max_steps",
        "answer": (
            "Research could not be completed "
            "within the maximum number of steps."
        ),
    }