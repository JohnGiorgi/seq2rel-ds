import random
from enum import Enum
from operator import itemgetter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from seq2rel_ds.common.schemas import PubtatorAnnotation, PubtatorCluster
from sklearn.model_selection import train_test_split

SEED = 13370
NUMPY_SEED = 1337

END_OF_REL_SYMBOL = "@EOR@"
COREF_SEP_SYMBOL = ";"


class TextSegment(str, Enum):
    title = "title"
    abstract = "abstract"
    both = "both"


def set_seeds() -> None:
    """Sets the random seeds of python and numpy for reproducible preprocessing."""
    random.seed(SEED)
    np.random.seed(NUMPY_SEED)


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def sort_by_offset(items: List[str], offsets: List[int], **kwargs) -> List[str]:
    """Returns `items`, sorted in ascending order according to `offsets`"""
    if len(items) != len(offsets):
        raise ValueError(f"len(items) ({len(items)}) != len(offsets) ({len(offsets)})")
    packed = list(zip(items, offsets))
    packed = sorted(packed, key=itemgetter(1), **kwargs)
    sorted_items, _ = zip(*packed)
    sorted_items = list(sorted_items)
    return sorted_items


def train_valid_test_split(
    data: Iterable[Any],
    train_size: float = 0.7,
    valid_size: float = 0.1,
    test_size: float = 0.2,
    **kwargs: Any,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Given an iterable (`data`), returns train, valid and test partitions of size `train_size`,
    `valid_size` and `test_size`. Optional kwargs are passed to `sklearn.model_selection.train_test_split`

    See https://datascience.stackexchange.com/a/53161 for details.
    """
    # Round to avoid precision errors.
    train, test = train_test_split(data, test_size=round(1 - train_size, 4), **kwargs)
    valid, test = train_test_split(test, test_size=test_size / (test_size + valid_size), **kwargs)
    return train, valid, test


def format_relation(ent_clusters: List[List[str]], ent_labels: List[str], rel_label: str) -> str:
    """Given an arbitrary number of coreferent mentions (`ent_clusters`), a label for each of those
    mentions (`ent_labels`) and a label for the relation (`rel_label`) returns a formatted string
    that can be used to train a seq2rel model.
    """
    formatted_rel = f"@{rel_label.strip().upper()}@"
    for ents, label in zip(ent_clusters, ent_labels):
        ents = sanitize_text(f"{COREF_SEP_SYMBOL} ".join(ents), lowercase=True)
        formatted_rel += f" {ents} @{label.strip().upper()}@"
    formatted_rel += f" {END_OF_REL_SYMBOL}"
    return formatted_rel


def parse_pubtator(
    pubtator_content: str,
    text_segment: TextSegment = TextSegment.both,
    skip_malformed: bool = False,
) -> Dict[str, PubtatorAnnotation]:
    """Parses a PubTator format string (`pubtator_content`) returning a highly structured,
    dictionary keyed by PMID.
    """
    # Get a list of PubTator annotations
    articles = pubtator_content.strip().split("\n\n")

    # Parse the annotations, producing a highly structured output
    parsed = {}
    for article in articles:
        article = article.strip().split("\n")
        title, abstract, annotations = article[0], article[1], article[2:]
        pmid, title = title.split("|t|")
        abstract = abstract.split("|a|")[-1]
        # Some very basic preprocessing
        title = sanitize_text(title)
        abstract = sanitize_text(abstract)

        # We may want to experiement with different text sources
        if text_segment.value == "both":
            text = f"{title} {abstract}"
        elif text_segment.value == "title":
            text = title
        else:
            text = abstract

        parsed[pmid] = PubtatorAnnotation(text=text)

        for ann in annotations:
            ann = ann.strip().split("\t")

            if len(ann) >= 6:
                if len(ann) == 6:
                    _, start, end, texts, label, uids = ann
                elif len(ann) == 7:
                    _, start, end, _, label, uids, texts = ann
                start, end = int(start), int(end)

                # Ignore this annotation if it is not in the chosen text segment
                section = "title" if int(start) < len(title) else "abstract"
                if section != text_segment.value and text_segment.value != "both":
                    continue

                # In at least one corpus (BC5CDR), the annotators include annotations for the
                # individual entities in a compound entity. So we deal with that here.
                # Note that the start & end indicies will no longer be exactly correct, but are
                # be close enough for our purposes of sorting entities by order of appearence.
                texts, uids = texts.split("|"), uids.split("|")

                for text, uid in zip(texts, uids):
                    # All entities are lowercased here to simplify the logic downstream.
                    # They will be lowercased by the copy mechanism anyways.
                    text = sanitize_text(text, lowercase=True)

                    if uid in parsed[pmid].clusters:
                        # Don't retain duplicates
                        if text not in parsed[pmid].clusters[uid].ents:
                            parsed[pmid].clusters[uid].ents.append(text)
                            parsed[pmid].clusters[uid].offsets.append((start, end))
                    else:
                        parsed[pmid].clusters[uid] = PubtatorCluster(
                            ents=[text], offsets=[(start, end)], label=label
                        )
            elif len(ann) == 4:  # this is a relation
                _, label, uid_1, uid_2 = ann
                if uid_1 in parsed[pmid].clusters and uid_2 in parsed[pmid].clusters:
                    parsed[pmid].relations.append((uid_1, uid_2, label))
            # For some cases (like distant supervision) it is convient to skip any annotations
            # that are malformed.
            else:
                if skip_malformed:
                    continue
                else:
                    err_msg = "Found an annotation with an unexpected number of columns: {}"
                    raise ValueError(err_msg.format(format(" ".join(ann))))

    return parsed
