import re

from hypothesis import given
from hypothesis.strategies import booleans, text

from seq2rel_ds.common import text_utils


def test_findall() -> None:
    # Test a simple case where there is no match.
    string = "The substring is not here!"
    substring = "test"
    assert list(text_utils.findall(string, substring)) == []
    # Test a simple case where there is one match.
    string = "The substring is here! test"
    substring = "test"
    assert list(text_utils.findall(string, substring)) == [23]
    # Test a simple case where there is more than one (non-overlapping) match.
    string = "The substring is here twice! test test"
    substring = "test"
    assert list(text_utils.findall(string, substring)) == [29, 34]
    # Test a simple case where there is more than one (overlapping) match.
    string = "GATATATGCATATACTT"
    substring = "ATAT"
    assert list(text_utils.findall(string, substring)) == [1, 3, 9]


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized_text = text_utils.sanitize_text(text, lowercase=lowercase)

    # There should be no cases of multiple spaces or tabs
    assert re.search(r"[ ]{2,}", sanitized_text) is None
    assert "\t" not in sanitized_text
    # The beginning and end of the string should be stripped of whitespace
    assert not sanitized_text.startswith(("\n", " "))
    assert not sanitized_text.endswith(("\n", " "))
    # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
    # We don't particularly care about this, and it breaks this check.
    # Only run if the generated text can be lowercased.
    if lowercase and text.lower().islower():
        assert all(not char.isupper() for char in sanitized_text)
