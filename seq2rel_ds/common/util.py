import random
from typing import Any, Iterable, List, Tuple

import numpy as np

from fuzzywuzzy import process
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


def fuzzy_match(
    queries: List[str], choices: List[str], return_longest: bool = True, **kwargs: Any
) -> str:
    """Returns the best fuzzy match from `choices` given `queries`. If `return_longest`, returns the
    longest best string match. Optional **kwargs are passed to `fuzzywuzzy.process.extractBests`.
    """
    best_match = ""
    fuzzy_matches = []
    for query in queries:
        # BioGRID uses "-" to denote empty values (e.g. no synonyms) causing fuzzywuzzy
        # to complain. Skip these values to prevent the warning from cluttering our output.
        if query == "-":
            continue
        fuzzy_matches.extend(process.extractBests(query, choices, **kwargs))
    if fuzzy_matches:
        if return_longest:
            best_match = max(fuzzy_matches, key=lambda x: len(x[0]))[0]
        else:
            best_match = fuzzy_matches[0][0]
    return best_match


def train_valid_test_split(
    data: Iterable[Any],
    train_size: int = 0.7,
    valid_size: int = 0.1,
    test_size: int = 0.2,
    **kwargs: Any,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Given an iterable (`data`), returns train, valid and test partitions of size `train_size`,
    `valid_size` and `test_size`. Optional kwargs are passed to `sklearn.model_selection.train_test_split`
    """
    # https://datascience.stackexchange.com/a/53161
    X_train, X_test = train_test_split(data, test_size=1 - train_size, **kwargs)
    X_valid, X_test = train_test_split(
        X_test, test_size=test_size / (test_size + valid_size), **kwargs
    )
    return X_train, X_valid, X_test
