# Public functions #

from typing import Iterable


def findall(string: str, substring: str) -> Iterable[int]:
    """Find all overlapping occurrences of `substring` in a `string`.
    Taken from: https://stackoverflow.com/a/34445090/6578628
    """
    i = string.find(substring)
    while i != -1:
        yield i
        i = string.find(substring, i + 1)


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text
