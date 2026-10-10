"""Plain terminal text suitable for redirected logs."""

from enum import Enum


class Color(str, Enum):
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"

    def __call__(self, text):
        return str(text)
