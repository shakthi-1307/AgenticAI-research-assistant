import time

from groq import Groq, RateLimitError

from .config import GROQ_API_KEY, MODEL


client = Groq(api_key=GROQ_API_KEY)

MAX_RETRIES = 2
RETRY_DELAY = 5


# =========================================================
# Runtime LLM usage statistics
# =========================================================

LLM_STATS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,

    "by_component": {}
}


# =========================================================
# Statistics
# =========================================================

def reset_llm_stats():
    """
    Reset all LLM usage statistics.
    """

    LLM_STATS["calls"] = 0
    LLM_STATS["prompt_tokens"] = 0
    LLM_STATS["completion_tokens"] = 0
    LLM_STATS["total_tokens"] = 0
    LLM_STATS["by_component"] = {}


def get_llm_stats():
    """
    Return a copy of the current LLM statistics.
    """

    return {
        "calls": LLM_STATS["calls"],
        "prompt_tokens": LLM_STATS["prompt_tokens"],
        "completion_tokens": LLM_STATS["completion_tokens"],
        "total_tokens": LLM_STATS["total_tokens"],
        "by_component": {
            component: stats.copy()
            for component, stats
            in LLM_STATS["by_component"].items()
        }
    }


# =========================================================
# Main LLM Gateway
# =========================================================

def call_llm(
    messages,
    tools=None,
    tool_choice=None,
    response_format=None,
    max_tokens=800,
    component="unknown",
):
    """
    Centralized LLM gateway.

    Responsibilities:
    - Call Groq
    - Handle rate-limit retries
    - Track token usage
    - Track latency
    - Track usage by component

    component examples:
    - planner
    - research_worker
    - summarizer
    - final_synthesis
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

            # -------------------------------------------------
            # Measure latency
            # -------------------------------------------------

            start_time = time.time()

            response = client.chat.completions.create(
                **kwargs
            )

            latency = time.time() - start_time

            # -------------------------------------------------
            # Extract token usage
            # -------------------------------------------------

            usage = response.usage

            prompt_tokens = getattr(
                usage,
                "prompt_tokens",
                0
            )

            completion_tokens = getattr(
                usage,
                "completion_tokens",
                0
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                0
            )

            # =================================================
            # Global statistics
            # =================================================

            LLM_STATS["calls"] += 1

            LLM_STATS["prompt_tokens"] += prompt_tokens

            LLM_STATS["completion_tokens"] += completion_tokens

            LLM_STATS["total_tokens"] += total_tokens

            # =================================================
            # Component statistics
            # =================================================

            if component not in LLM_STATS["by_component"]:

                LLM_STATS["by_component"][component] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

            component_stats = (
                LLM_STATS["by_component"][component]
            )

            component_stats["calls"] += 1

            component_stats["prompt_tokens"] += (
                prompt_tokens
            )

            component_stats["completion_tokens"] += (
                completion_tokens
            )

            component_stats["total_tokens"] += (
                total_tokens
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            print(
                f"LLM call #{LLM_STATS['calls']} | "
                f"component: {component} | "
                f"tokens: {total_tokens} | "
                f"latency: {latency:.2f}s"
            )

            return response.choices[0].message

        # =====================================================
        # Rate-limit handling
        # =====================================================

        except RateLimitError as error:

            if attempt >= MAX_RETRIES:
                raise error

            wait_time = RETRY_DELAY * (attempt + 1)

            print(
                f"LLM rate limit reached. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        # =====================================================
        # Other errors
        # =====================================================

        except Exception:
            raise