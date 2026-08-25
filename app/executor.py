import asyncio
import json

from .tools import execute_tool_async


MAX_TOOL_RETRIES = 2
MAX_TOOL_RESULT_CHARS = 12000


def limit_tool_result(result):

    if not isinstance(result, str):
        result = str(result)

    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result

    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + "\n\n"
        "[Tool result truncated because it was too large.]"
    )


async def execute_tool_with_retry(
    name,
    arguments,
):

    for attempt in range(
        MAX_TOOL_RETRIES + 1
    ):

        try:

            print(
                f"Executing {name} "
                f"(attempt {attempt + 1})"
            )

            result = await execute_tool_async(
                name,
                arguments,
            )

            return limit_tool_result(
                result
            )

        except Exception as error:

            print(
                f"Tool '{name}' failed: {error}"
            )

            if attempt == MAX_TOOL_RETRIES:

                return {
                    "error": (
                        f"Tool '{name}' failed after "
                        f"{MAX_TOOL_RETRIES + 1} attempts."
                    )
                }

            await asyncio.sleep(1)


async def execute_tool_call(
    tool_call
):

    name = tool_call.function.name

    try:

        arguments = json.loads(
            tool_call.function.arguments
        )

        print(f"Tool: {name}")
        print(f"Arguments: {arguments}")

        result = await execute_tool_with_retry(
            name,
            arguments,
        )

    except Exception as error:

        arguments = {}

        result = {
            "error": str(error)
        }

    return {
        "tool_call": tool_call,
        "name": name,
        "arguments": arguments,
        "result": result,
    }


async def execute_tool_calls(
    tool_calls
):

    """
    Execute multiple independent tool calls
    concurrently.
    """

    tasks = [
        execute_tool_call(
            tool_call
        )
        for tool_call in tool_calls
    ]

    return await asyncio.gather(
        *tasks
    )