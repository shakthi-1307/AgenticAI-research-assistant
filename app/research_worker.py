import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool_async
from .summarizer import summarize_content


client = Groq(api_key=GROQ_API_KEY)

MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 12000


def prepare_tool_result(result):
    """
    Convert a tool result to JSON and prevent
    extremely large results from entering the
    LLM context.
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
    """
    Convert results from previous tasks into
    context that can be given to the research worker.
    """

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


async def research_task(task, dependency_results=None):

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
  "input", or any other argument name.
- Always follow the exact argument schema.

RESEARCH PROCESS:

1. Understand the research question.
2. Check whether previous task results are available.
3. Use previous results when they are relevant.
4. Search the web when more information is needed.
5. Fetch useful pages when detailed information
   is required.
6. Evaluate the information you have collected.
7. Search again only when:
   - important information is missing,
   - sources conflict,
   - or stronger evidence is needed.
8. Stop researching once you have enough reliable
   information to answer the question.

Do NOT continue searching simply because another
step is available.

The maximum number of research steps is a
safety limit, not a target.

When you have enough information, stop using tools
and provide a concise factual answer.

Do not invent facts.

When previous task results are provided, treat them
as useful context, but verify them when necessary.
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
                "error": str(error)
            }

        message = response.choices[0].message

        # --------------------------------------------------
        # Agent decided that it has enough information
        # --------------------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content or ""
            }

        # --------------------------------------------------
        # Execute tool calls
        # --------------------------------------------------

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
                        "content": message.content or "",
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The tool arguments for {name} "
                            "were invalid JSON. "
                            "Please retry using valid JSON "
                            "and the exact required schema."
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

            # --------------------------------------------------
            # Prevent duplicate searches
            # --------------------------------------------------

            if name == "web_search":

                query = arguments.get("query")

                if not query:

                    print(
                        "web_search called without query"
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "web_search requires the "
                                'argument {"query": "..."}'
                            )
                        }
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

            # --------------------------------------------------
            # Prevent duplicate page fetches
            # --------------------------------------------------

            if name == "fetch_page":

                url = arguments.get("url")

                if not url:

                    print(
                        "fetch_page called without URL"
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "fetch_page requires the "
                                'argument {"url": "..."}'
                            )
                        }
                    )

                    continue

                if url in used_urls:

                    print(
                        "Skipping duplicate URL"
                    )

                    continue

                used_urls.add(url)

            # --------------------------------------------------
            # Execute tool
            # --------------------------------------------------

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
                        f"Tool execution failed: {error}"
                    )
                }

            # --------------------------------------------------
            # Summarize fetched webpages
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
                        question=question
                    )

                    result = {
                        "url": result.get("url"),
                        "summary": summary
                    }

                except Exception as error:

                    print(
                        f"Could not summarize webpage: "
                        f"{error}"
                    )

                    result = {
                        "url": result.get("url"),
                        "error": (
                            "Could not summarize webpage: "
                            f"{error}"
                        )
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
                                )
                            }
                        }
                    ]
                }
            )

            # --------------------------------------------------
            # Add tool result to conversation
            # --------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": prepare_tool_result(
                        result
                    )
                }
            )

    # ------------------------------------------------------
    # Maximum step limit reached
    # ------------------------------------------------------

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