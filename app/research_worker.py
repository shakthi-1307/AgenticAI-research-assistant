import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .tools import TOOLS, execute_tool


client = Groq(
    api_key=GROQ_API_KEY
)

MAX_RESEARCH_STEPS = 3
MAX_TOOL_RESULT_CHARS = 8000
MAX_MESSAGES = 8


def clean_tool_result(result):
    """
    Keep tool results small before
    sending them back to the LLM.
    """

    content = json.dumps(
        result,
        ensure_ascii=False
    )

    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + "\n...[truncated]"
        )

    return content


def trim_messages(messages):
    """
    Keep the system prompt and the
    most recent messages.
    """

    if len(messages) <= MAX_MESSAGES:
        return messages

    system_message = messages[0]

    recent_messages = messages[
        -(MAX_MESSAGES - 1):
    ]

    return [
        system_message,
        *recent_messages
    ]


def research_task(task):

    messages = [
        {
            "role": "system",
            "content": """
You are a research worker.

Your job is to answer ONE research question.

You have two possible actions:

1. SEARCH
   Use web_search to find information.

2. FINISH
   Provide the final answer when enough
   evidence has been collected.

IMPORTANT RULES:

- Start with web_search for factual questions.
- After receiving search results, inspect them.
- If the results are sufficient to answer,
  FINISH immediately.
- Do not perform another search if the existing
  results already contain the answer.
- Never repeat the same search query.
- Do not search just because another iteration
  is available.
- For simple questions, one search is normally
  enough.

When you finish, give a concise answer based
only on the gathered information.
""",
        },
        {
            "role": "user",
            "content": (
                "Research question:\n"
                + task["task"]
            ),
        },
    ]

    previous_queries = set()

    # NEW: track worker execution
    tool_calls = []
    errors = []
    sources = []

    for step in range(
        MAX_RESEARCH_STEPS
    ):

        print(
            f"Research worker step "
            f"{step + 1}"
        )

        messages = trim_messages(
            messages
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000,
        )

        message = response.choices[0].message

        # -------------------------------------
        # FINISHED
        # -------------------------------------

        if not message.tool_calls:

            return {
                "status": "complete",
                "answer": message.content,
                "sources": sources,
                "tool_calls": tool_calls,
                "steps_used": step + 1,
                "errors": errors,
            }

        messages.append(message)

        # -------------------------------------
        # EXECUTE TOOL CALLS
        # -------------------------------------

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            # ---------------------------------
            # Prevent repeated searches
            # ---------------------------------

            if name == "web_search":

                query = arguments["query"]

                if query in previous_queries:

                    print(
                        "Repeated search detected."
                    )

                    errors.append({
                        "type": "repeated_search",
                        "query": query,
                    })

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call.id
                            ),
                            "name": name,
                            "content": (
                                "This search query "
                                "was already executed. "
                                "Do not repeat it. "
                                "Use the existing results "
                                "and finish the task."
                            ),
                        }
                    )

                    continue

                previous_queries.add(query)

            print(
                f"Research worker tool: "
                f"{name}"
            )

            result = execute_tool(
                name,
                arguments
            )

            # ---------------------------------
            # Track tool call
            # ---------------------------------

            tool_calls.append({
                "tool": name,
                "arguments": arguments,
                "step": step + 1,
            })

            # ---------------------------------
            # Track sources
            # ---------------------------------

            if name == "web_search":

                if isinstance(result, list):

                    for item in result:

                        if isinstance(item, dict):

                            url = item.get("url")

                            if url:
                                sources.append(url)

            print("\n--- TOOL RESULT ---")

            print(
                json.dumps(
                    result,
                    indent=2
                )[:5000]
            )

            print(
                "--- END TOOL RESULT ---\n"
            )

            # ---------------------------------
            # Clean result
            # ---------------------------------

            content = clean_tool_result(
                result
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "name": name,
                    "content": content,
                }
            )

    # -----------------------------------------
    # MAX STEPS
    # -----------------------------------------

    return {
        "status": "failed",
        "answer": (
            "Research worker could not "
            "complete the task within "
            "the allowed number of steps."
        ),
        "sources": sources,
        "tool_calls": tool_calls,
        "steps_used": MAX_RESEARCH_STEPS,
        "errors": errors,
    }