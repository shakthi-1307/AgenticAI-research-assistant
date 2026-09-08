import json

from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .research_worker import research_task
from .tools import execute_tool


client = Groq(api_key=GROQ_API_KEY)


async def execute_worker(task, dependency_results=None):

    dependency_results = dependency_results or []

    task_type = task["type"]

    # --------------------------------------------------
    # RESEARCH
    # --------------------------------------------------

    if task_type == "research":

        return await research_task(
            task,
            dependency_results
        )

    # --------------------------------------------------
    # WEATHER
    # --------------------------------------------------

    if task_type == "weather":

        city = task.get("city")

        if not city:
            return {
                "error": "Weather task has no city."
            }

        return execute_tool(
            "get_weather",
            {
                "city": city
            }
        )

    # --------------------------------------------------
    # CALCULATION
    # --------------------------------------------------

    if task_type == "calculation":

        expression = task.get("expression")

        # If there are no dependencies,
        # use the expression directly.
        if not dependency_results:

            if not expression:
                return {
                    "error": (
                        "Calculation task has no expression."
                    )
                }

            return execute_tool(
                "calculator",
                {
                    "expression": expression
                }
            )

        # --------------------------------------------------
        # Use dependency results to construct
        # an executable mathematical expression.
        # --------------------------------------------------

        dependency_context = json.dumps(
            dependency_results,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are a calculation assistant.

The user wants this calculation:

{task["task"]}

The planner provided this expression:

{expression or "No expression provided"}

Here are the results from previous tasks:

{dependency_context}

Your job is to determine the actual numerical
values needed from the previous results and
construct ONE valid mathematical expression.

Rules:

- Extract only numerical values relevant to
  the requested calculation.
- Replace placeholders such as POPULATION with
  the actual numerical value.
- Do not invent numbers.
- Do not include units or words.
- Use only numbers and these operators:
  + - * / ( )
- Return ONLY the mathematical expression.
- Do not explain anything.

Example:

Previous result:
"India population is 1,428,600,000"

Task:
"Convert India's population into millions"

Expression:
"POPULATION / 1000000"

Correct output:
1428600000 / 1000000
"""

        try:

            response = client.chat.completions.create(
                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only a valid mathematical "
                            "expression."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                tool_choice="none",
                max_tokens=200
            )

            resolved_expression = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as error:

            return {
                "error": (
                    f"Could not resolve calculation: "
                    f"{error}"
                )
            }

        print(
            f"Resolved calculation: "
            f"{resolved_expression}"
        )

        # --------------------------------------------------
        # Execute the resolved expression
        # --------------------------------------------------

        try:

            result = execute_tool(
                "calculator",
                {
                    "expression": resolved_expression
                }
            )

        except Exception as error:

            return {
                "error": (
                    f"Calculation failed: {error}"
                )
            }

        return {
            "expression": resolved_expression,
            "result": result
        }

    # --------------------------------------------------
    # UNKNOWN TASK
    # --------------------------------------------------

    return {
        "error": f"Unknown task type: {task_type}"
    }