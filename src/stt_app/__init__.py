"""Push-to-record speech-to-text terminal app."""


def main() -> None:
    from .ui import main as _main

    _main()


__all__ = ["main"]
