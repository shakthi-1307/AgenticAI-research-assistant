from dataclasses import dataclass, field


@dataclass
class AgentState:
    question: str = ""

    messages: list = field(default_factory=list)

    plan: list = field(default_factory=list)

    results: list = field(default_factory=list)

    iteration: int = 0

    final_answer: str | None = None

    # LLM usage for this run
    llm_usage: dict = field(
        default_factory=lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )