import json
import os

from .state import AgentState


CHECKPOINT_DIR = "checkpoints"


def get_checkpoint_path():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    return os.path.join(
        CHECKPOINT_DIR,
        "agent_state.json"
    )


def save_checkpoint(state):

    path = get_checkpoint_path()

    checkpoint_data = {
        "question": state.question,
        "messages": state.messages,
        "plan": state.plan,
        "results": state.results,
        "iteration": state.iteration,
        "final_answer": state.final_answer
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            checkpoint_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Checkpoint saved: {path}"
    )


def load_checkpoint():

    path = get_checkpoint_path()

    if not os.path.exists(path):

        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        state = AgentState(
            question=data.get(
                "question",
                ""
            ),
            messages=data.get(
                "messages",
                []
            ),
            plan=data.get(
                "plan",
                []
            ),
            results=data.get(
                "results",
                []
            ),
            iteration=data.get(
                "iteration",
                0
            ),
            final_answer=data.get(
                "final_answer"
            )
        )

        print(
            f"Checkpoint loaded: {path}"
        )

        return state

    except Exception as error:

        print(
            f"Could not load checkpoint: {error}"
        )

        return None


def delete_checkpoint():

    path = get_checkpoint_path()

    if os.path.exists(path):

        os.remove(path)

        print(
            "Checkpoint deleted."
        )