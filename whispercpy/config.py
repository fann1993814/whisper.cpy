from pathlib import Path


LIB_PATH: Path | None = None


def set_lib_path(path: str | Path | None) -> None:
    global LIB_PATH
    LIB_PATH = Path(path) if path is not None else None


def get_lib_path() -> Path:
    if LIB_PATH is not None:
        return LIB_PATH
    else:
        raise RuntimeError(
            "The whisper.cpp library path is not set. "
            "Please call `set_lib_path(path)` before using whispercpy."
        )
