import re

import numpy as np
from hypothesis import given
from hypothesis.strategies import booleans, text
from seq2rel_ds.common import util


def test_set_seeds() -> None:
    # As it turns out, it is quite hard to determine the current random
    # seed. I was able to determine how for numpy, but not python or tensorflow.
    # See: https://stackoverflow.com/a/49749486/6578628
    util.set_seeds()
    assert np.random.get_state()[1][0] == util.NUMPY_SEED


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized_text = util.sanitize_text(text, lowercase=lowercase)

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
