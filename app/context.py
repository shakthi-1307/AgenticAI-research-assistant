import json


MAX_CONTEXT_CHARS = 12000


def build_context(state):

    context = {
        "plan": state.plan,
        "working_memory": state.results,
    }

    serialized = json.dumps(
        context,
        indent=2,
        ensure_ascii=False,
    )

    if len(serialized) <= MAX_CONTEXT_CHARS:
        return serialized

    return (
        serialized[:MAX_CONTEXT_CHARS]
        + "\n\n"
        "[Context truncated.]"
    )