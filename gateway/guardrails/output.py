"""Checks on the model's answer before it goes back to the caller."""

WINDOW = 60
STEP = 20


def leaks_system_prompt(system_text: str, answer: str) -> bool:
    """True when the answer repeats a long verbatim stretch of the system prompt.

    A successful injection usually shows up here even if the input check missed it.
    """
    system = " ".join(system_text.split())
    if len(system) < WINDOW:
        return False
    flat = " ".join(answer.split())
    return any(system[i : i + WINDOW] in flat for i in range(0, len(system) - WINDOW + 1, STEP))
