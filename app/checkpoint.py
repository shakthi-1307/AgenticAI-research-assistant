import json
import os
import uuid

from .state import AgentState


CHECKPOINT_DIR = "checkpoints"


def create_run_id():
    """
    Create a unique ID for an agent execution.
    """

    return str(uuid.uuid4())


def get_checkpoint_path(run_id):
    """
    Return the checkpoint file for a specific run.
    """

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    return os.path.join(
        CHECKPOINT_DIR,
        f"{run_id}.json"
    )


def save_checkpoint(state, run_id):

    path = get_checkpoint_path(
        run_id
    )

    checkpoint_data = {
        "run_id": run_id,
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


def load_checkpoint(run_id):

    path = get_checkpoint_path(
        run_id
    )

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
            f"Could not load checkpoint: "
            f"{error}"
        )

        return None


def delete_checkpoint(run_id):

    path = get_checkpoint_path(
        run_id
    )

    if os.path.exists(path):

        os.remove(path)

        print(
            f"Checkpoint deleted: {path}"
        )