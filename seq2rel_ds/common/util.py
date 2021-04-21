import random
from typing import Any, Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 13370
NUMPY_SEED = 1337


def set_seeds():
    """Sets the random seeds of python and numpy for reproducible preprocessing."""
    random.seed(SEED)
    np.random.seed(NUMPY_SEED)


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def train_valid_test_split(
    data: Iterable[Any],
    train_size: float = 0.7,
    valid_size: float = 0.1,
    test_size: float = 0.2,
    **kwargs: Any,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Given an iterable (`data`), returns train, valid and test partitions of size `train_size`,
    `valid_size` and `test_size`. Optional kwargs are passed to `sklearn.model_selection.train_test_split`
    """
    # https://datascience.stackexchange.com/a/53161
    train, test = train_test_split(data, test_size=1 - train_size, **kwargs)
    valid, test = train_test_split(test, test_size=test_size / (test_size + valid_size), **kwargs)
    return train, valid, test
