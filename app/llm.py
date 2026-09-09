import time

from groq import Groq, RateLimitError

from .config import GROQ_API_KEY, MODEL


client = Groq(api_key=GROQ_API_KEY)

MAX_RETRIES = 2
RETRY_DELAY = 5


def call_llm(
    messages,
    tools=None,
    tool_choice=None,
    response_format=None,
    max_tokens=800
):
    """
    Centralized LLM interface.

    All components should use this function
    instead of calling Groq directly.
    """

    for attempt in range(MAX_RETRIES + 1):

        try:

            kwargs = {
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if tools is not None:
                kwargs["tools"] = tools

            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(
                **kwargs
            )

            return response.choices[0].message

        except RateLimitError as error:

            if attempt >= MAX_RETRIES:
                raise error

            wait_time = RETRY_DELAY * (attempt + 1)

            print(
                f"LLM rate limit reached. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        except Exception:
            raise