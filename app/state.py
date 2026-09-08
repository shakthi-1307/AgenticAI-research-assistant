from dataclasses import dataclass, field


@dataclass
class AgentState:

    question: str = ""

    messages: list = field(
        default_factory=list
    )

    plan: list = field(
        default_factory=list
    )

    results: list = field(
        default_factory=list
    )

    iteration: int = 0

    final_answer: str | None = None