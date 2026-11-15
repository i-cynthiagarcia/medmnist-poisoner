"""Runtime type checks for values read from Flower configuration records."""


def require_config_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Configuration value {name!r} must be a string.")
    return value


def require_config_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Configuration value {name!r} must be an integer.")
    return value


def require_config_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Configuration value {name!r} must be numeric.")
    return float(value)
