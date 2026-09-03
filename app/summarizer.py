from groq import Groq

from .config import GROQ_API_KEY, MODEL


client = Groq(
    api_key=GROQ_API_KEY
)


MAX_INPUT_CHARS = 12000


def summarize_content(
    content: str,
    question: str,
):
    """
    Compress webpage content into
    concise evidence relevant to the
    research question.
    """

    content = content[:MAX_INPUT_CHARS]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an evidence summarizer.

Summarize the provided webpage content
only for the given research question.

Rules:

- Keep only information relevant to the question.
- Do not invent information.
- Ignore navigation, menus, scripts, CSS,
  metadata, advertisements, and unrelated text.
- Preserve important facts.
- Be concise.
- Return plain text.
""",
            },
            {
                "role": "user",
                "content": (
                    f"Research question:\n"
                    f"{question}\n\n"
                    f"Webpage content:\n"
                    f"{content}"
                ),
            },
        ],
        tool_choice="none",
        max_tokens=800,
    )

    return response.choices[0].message.content