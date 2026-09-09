import json

from .llm import call_llm
from .research_worker import research_task
from .tools import execute_tool_async


async def execute_worker(task, dependency_results=None):
    """
    Execute a task using the appropriate specialized worker.
    """

    dependency_results = dependency_results or []

    task_type = task["type"].upper()

    # ---------------------------------------------------------
    # RESEARCH
    # ---------------------------------------------------------
    if task_type == "RESEARCH":
        return await research_task(
            task,
            dependency_results=dependency_results
        )

    # ---------------------------------------------------------
    # WEATHER
    # ---------------------------------------------------------
    elif task_type == "WEATHER":
        city = task.get("city")

        if not city:
            return {
                "status": "failed",
                "error": "Weather task is missing a city."
            }

        try:
            result = await execute_tool_async(
                "get_weather",
                {"city": city}
            )

            return {
                "status": "complete",
                "data": result
            }

        except Exception as error:
            return {
                "status": "failed",
                "error": str(error)
            }

    # ---------------------------------------------------------
    # CALCULATION
    # ---------------------------------------------------------
    elif task_type == "CALCULATION":

        expression = task.get("expression")

        # -----------------------------------------------------
        # Direct calculation
        # -----------------------------------------------------
        if not dependency_results:

            if not expression:
                return {
                    "status": "failed",
                    "error": "Calculation task is missing an expression."
                }

            try:
                result = await execute_tool_async(
                    "calculator",
                    {"expression": expression}
                )

                return {
                    "status": "complete",
                    "expression": expression,
                    "result": result
                }

            except Exception as error:
                return {
                    "status": "failed",
                    "error": str(error)
                }

        # -----------------------------------------------------
        # Calculation based on previous research
        # -----------------------------------------------------
        dependency_context = json.dumps(
            dependency_results,
            indent=2,
            ensure_ascii=False
        )

        messages = [
            {
                "role": "system",
                "content": """
You extract numerical values from research results.

Return ONLY valid JSON in exactly this format:

{
  "value_a": number,
  "value_b": number
}

Rules:
- value_a must be the India population.
- value_b must be the China population.
- Use only numbers explicitly supported by the research results.
- Do not include commas in numbers.
- Do not include explanations.
- Do not include markdown.
"""
            },
            {
                "role": "user",
                "content": (
                    "Extract the required values from these research results:\n\n"
                    f"{dependency_context}"
                )
            }
        ]

        try:
            message = call_llm(
                messages=messages,
                tools=None,
                tool_choice="none",
                max_tokens=200
            )

            extracted = json.loads(message.content)

            value_a = extracted.get("value_a")
            value_b = extracted.get("value_b")

            if not isinstance(value_a, (int, float)):
                raise ValueError("value_a is not numeric.")

            if not isinstance(value_b, (int, float)):
                raise ValueError("value_b is not numeric.")

            if value_a <= 0 or value_b <= 0:
                raise ValueError("Extracted values must be positive.")

            # Deterministic calculation.
            calculation_expression = (
                f"abs({value_a}-{value_b})/"
                f"(({value_a}+{value_b})/2)*100"
            )

            result = await execute_tool_async(
                "calculator",
                {"expression": calculation_expression}
            )

            return {
                "status": "complete",
                "value_a": value_a,
                "value_b": value_b,
                "expression": calculation_expression,
                "result": result
            }

        except Exception as error:
            return {
                "status": "failed",
                "error": str(error)
            }

    # ---------------------------------------------------------
    # UNKNOWN TASK TYPE
    # ---------------------------------------------------------
    else:
        return {
            "status": "failed",
            "error": f"Unknown task type: {task_type}"
        }