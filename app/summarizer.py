from .llm import call_llm


MAX_INPUT_CHARS = 12000


def summarize_content(content: str, question: str):
    """
    Summarize webpage content specifically for the research question.
    """

    content = content[:MAX_INPUT_CHARS]

    messages = [
        {
            "role": "system",
            "content": """
You are an evidence summarizer.

Summarize the provided webpage content only for the given research question.

Rules:
- Keep only information relevant to the question.
- Do not invent information.
- Do not make unsupported assumptions.
- Ignore navigation, menus, scripts, CSS, metadata,
  advertisements, and unrelated text.
- Preserve important facts, numbers, names, and dates.
- Be concise.
- Return plain text.
"""
        },
        {
            "role": "user",
            "content": (
                f"Research question:\n{question}\n\n"
                f"Webpage content:\n{content}"
            )
        }
    ]

    message = call_llm(
        messages=messages,
        tools=None,
        tool_choice="none",
        max_tokens=800,
    )

    return message.content or ""