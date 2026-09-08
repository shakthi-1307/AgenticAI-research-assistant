import json
from groq import Groq

from .config import GROQ_API_KEY, MODEL
from .research_worker import research_task
from .tools import execute_tool


client = Groq(api_key=GROQ_API_KEY)


async def execute_worker(task, dependency_results=None):
    dependency_results = dependency_results or []
    task_type = task["type"]

    # ---------------------------------------------------------
    # RESEARCH
    # ---------------------------------------------------------
    if task_type == "research":
        return await research_task(task, dependency_results)

    # ---------------------------------------------------------
    # WEATHER
    # ---------------------------------------------------------
    if task_type == "weather":

        city = task.get("city")

        if not city:
            return {
                "status": "failed",
                "error": "Weather task has no city."
            }

        result = execute_tool(
            "get_weather",
            {"city": city}
        )

        if isinstance(result, dict) and "error" in result:
            return {
                "status": "failed",
                "error": result["error"]
            }

        return {
            "status": "complete",
            "data": result
        }

    # ---------------------------------------------------------
    # CALCULATION
    # ---------------------------------------------------------
    if task_type == "calculation":

        expression = task.get("expression")

        # -----------------------------------------------------
        # Direct calculation
        # -----------------------------------------------------

        if not dependency_results:

            if not expression:
                return {
                    "status": "failed",
                    "error": "Calculation task has no expression."
                }

            result = execute_tool(
                "calculator",
                {"expression": expression}
            )

            if isinstance(result, dict) and "error" in result:
                return {
                    "status": "failed",
                    "error": result["error"]
                }

            return {
                "status": "complete",
                "expression": expression,
                "result": result
            }

        # -----------------------------------------------------
        # Dependency-based calculation
        # -----------------------------------------------------

        dependency_context = json.dumps(
            dependency_results,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You extract numerical values for a calculation.

Calculation requested:
{task["task"]}

Previous task results:
{dependency_context}

Extract the TWO numerical population values
needed for this calculation:

- value_a = India's population
- value_b = China's population

Rules:
- Use only numbers explicitly present in the previous results.
- Do not estimate.
- Do not invent numbers.
- Ignore unrelated numbers such as birth rate,
  death rate, growth rate, age, etc.
- Return ONLY valid JSON.
- Use this exact format:

{{
  "value_a": 1451000000,
  "value_b": 1408280000
}}
"""

        # -----------------------------------------------------
        # Ask LLM to extract values
        # -----------------------------------------------------

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the requested numerical values "
                            "and return only valid JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_object"
                },
                max_tokens=200
            )

            raw_content = (
                response.choices[0].message.content or ""
            ).strip()

        except Exception as error:

            return {
                "status": "failed",
                "error": (
                    f"Could not extract calculation values: {error}"
                )
            }

        if not raw_content:

            return {
                "status": "failed",
                "error": "Calculation extractor returned empty output."
            }

        # -----------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------

        try:

            extracted = json.loads(raw_content)

        except json.JSONDecodeError as error:

            return {
                "status": "failed",
                "error": (
                    f"Calculation extractor returned invalid JSON: "
                    f"{error}"
                )
            }

        # -----------------------------------------------------
        # Validate extracted values
        # -----------------------------------------------------

        value_a = extracted.get("value_a")
        value_b = extracted.get("value_b")

        if value_a is None or value_b is None:

            return {
                "status": "failed",
                "error": (
                    "Calculation extractor did not return "
                    "both required values."
                )
            }

        try:

            value_a = float(value_a)
            value_b = float(value_b)

        except (TypeError, ValueError):

            return {
                "status": "failed",
                "error": "Extracted values are not valid numbers."
            }

        if value_a <= 0 or value_b <= 0:

            return {
                "status": "failed",
                "error": "Extracted population values must be positive."
            }

        # -----------------------------------------------------
        # Deterministic calculation
        # -----------------------------------------------------

        # Percentage difference:
        #
        # |A - B|
        # ----------- × 100
        # (A + B) / 2

        expression = (
            f"({abs(value_a - value_b)})"
            f"/(({value_a}+{value_b})/2)*100"
        )

        print(
            f"Extracted values: "
            f"India={value_a}, China={value_b}"
        )

        print(
            f"Resolved calculation: {expression}"
        )

        # -----------------------------------------------------
        # Execute calculator
        # -----------------------------------------------------

        try:

            result = execute_tool(
                "calculator",
                {
                    "expression": expression
                }
            )

        except Exception as error:

            return {
                "status": "failed",
                "error": f"Calculation failed: {error}"
            }

        if isinstance(result, dict) and "error" in result:

            return {
                "status": "failed",
                "error": result["error"]
            }

        return {
            "status": "complete",
            "values": {
                "india": value_a,
                "china": value_b
            },
            "formula": (
                "|India - China| / "
                "((India + China) / 2) × 100"
            ),
            "expression": expression,
            "result": result
        }

    # ---------------------------------------------------------
    # UNKNOWN TASK
    # ---------------------------------------------------------

    return {
        "status": "failed",
        "error": f"Unknown task type: {task_type}"
    }