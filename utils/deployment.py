import os


def rapid_labeling_enabled() -> bool:
    """Read the deployment's Rapid Labeling switch independently of session state."""
    return os.environ.get("MIG_ENABLE_RAPID_LABELING", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
