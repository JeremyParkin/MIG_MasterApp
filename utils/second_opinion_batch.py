from __future__ import annotations


def prepare_second_opinion_batch_size(
    session_state,
    *,
    input_key: str,
    recommended_batch: int,
    available_count: int,
    refresh_recommendation: bool,
    fallback_batch_size: int = 10,
) -> int:
    """Seed a new recommendation without overwriting an in-cycle user choice."""
    available = max(0, int(available_count))
    if available == 0:
        return 0

    if refresh_recommendation:
        session_state.pop(input_key, None)

    if input_key not in session_state:
        proposed = int(recommended_batch) if int(recommended_batch) > 0 else int(fallback_batch_size)
        session_state[input_key] = min(available, max(1, proposed))
    else:
        session_state[input_key] = min(available, max(1, int(session_state[input_key])))

    return int(session_state[input_key])
