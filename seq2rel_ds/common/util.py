import random
from enum import Enum
from operator import itemgetter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from seq2rel_ds.common.schemas import PubtatorAnnotation, PubtatorCluster, PartitionStatistics
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
    size_sum = train_size + valid_size + test_size
    if size_sum != 1.0:
        raise ValueError(f"train_size, valid_size and test_size must sum to one. Got {size_sum}.")
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


def sort_entity_annotations(annotations: List[str]) -> List[str]:
    """Sort PubTator entity annotations by order of first appearence."""

    # We only sort the entities, so we have to seperate them from the relations,
    # perform the sort and then join everything together.
    ents = [ann for ann in annotations if len(ann.split("\t")) != 4]
    rels = [ann for ann in annotations if len(ann.split("\t")) == 4]
    sorted_ents = sorted(ents, key=lambda x: int(x.split("\t")[2]))
    return sorted_ents + rels


def parse_pubtator(
    pubtator_content: str,
    text_segment: TextSegment = TextSegment.both,
    sort_ents: bool = True,
    skip_malformed: bool = False,
) -> Dict[str, PubtatorAnnotation]:
    """Parses a PubTator format string (`pubtator_content`) returning a highly structured,
    dictionary-like object keyed by PMID.

    # Parameters

    pubtator_content : `str`
        A string containing one or more articles in the PubTator format.
    text_segment : `TextSegment`, optional (default = `TextSegment.both`)
        Which segment of the text we should consider. Valid values are `TextSegment.title`,
        `TextSegment.abstract` or `TextSegment.both`.
    sort_ents : bool, optional (default = `True`)
        Whether entities should be sorted by order of first appearence. This useful for traditional
        seq2seq models that use an order-sensitive loss function, like negative log likelihood.
    skip_malformed : bool, optional (default = `False`)
        True if we should ignore malformed annotations that cannot be parsed. This is useful in
        some cases, like when we are generating data using distant supervision.
    """
    # Get a list of PubTator annotations
    articles = pubtator_content.strip().split("\n\n")

    # Parse the annotations, producing a highly structured output
    parsed = {}
    for article in articles:
        split_article = article.strip().split("\n")
        title, abstract, annotations = split_article[0], split_article[1], split_article[2:]
        if sort_ents:
            annotations = sort_entity_annotations(annotations)
        pmid, title = title.split("|t|")
        abstract = abstract.split("|a|")[-1]
        # Some very basic preprocessing
        title = sanitize_text(title)
        abstract = sanitize_text(abstract)

        # We may want to experiement with different text sources
        if text_segment.value == "both":
            text = f"{title} {abstract}" if abstract else title
        elif text_segment.value == "title":
            text = title
        else:
            # In at least one corpus (GDA), there is a title but no abstract.
            # we handle that by ignoring it when text_segment is "both", and
            # raising an error when it is "abstract"
            if not abstract:
                msg = f"text_segment was {text_segment.value} but no abstract was found"
                raise ValueError(msg)
            text = abstract

        parsed[pmid] = PubtatorAnnotation(text=text)

        for ann in annotations:
            split_ann = ann.strip().split("\t")

            if len(split_ann) >= 6:
                if len(split_ann) == 6:
                    _, start, end, texts, label, uids = split_ann
                elif len(split_ann) == 7:
                    _, start, end, _, label, uids, texts = split_ann
                start, end = int(start), int(end)  # type: ignore

                # Ignore this annotation if it is not in the chosen text segment
                section = "title" if int(start) < len(title) else "abstract"
                if section != text_segment.value and text_segment.value != "both":
                    continue

                # In at least one corpus (BC5CDR), the annotators include annotations for the
                # individual entities in a compound entity. So we deal with that here.
                # Note that the start & end indicies will no longer be exactly correct, but are
                # be close enough for our purposes of sorting entities by order of appearence.
                texts, uids = texts.split("|"), uids.split("|")  # type: ignore

                for text, uid in zip(texts, uids):
                    # All entities are lowercased here to simplify the logic downstream.
                    # They will be lowercased by the copy mechanism anyways.
                    text = sanitize_text(text, lowercase=True)

                    if uid in parsed[pmid].clusters:
                        # Don't retain duplicate text...
                        if text not in parsed[pmid].clusters[uid].ents:
                            parsed[pmid].clusters[uid].ents.append(text)
                        # ...but do retain offsets of all mentions
                        parsed[pmid].clusters[uid].offsets.append((start, end))
                    else:
                        parsed[pmid].clusters[uid] = PubtatorCluster(
                            ents=[text], offsets=[(start, end)], label=label
                        )
            elif len(split_ann) == 4:  # this is a relation
                _, label, uid_1, uid_2 = split_ann
                if uid_1 in parsed[pmid].clusters and uid_2 in parsed[pmid].clusters:
                    parsed[pmid].relations.append((uid_1, uid_2, label))
            # For some cases (like distant supervision) it is convient to
            # skip annotations that are malformed.
            else:
                if skip_malformed:
                    continue
                else:
                    err_msg = f"Found an annotation with an unexpected number of columns: {ann}"
                    raise ValueError(err_msg)

    return parsed


def pubtator_to_seq2rel(pubtator_annotations: Dict[str, PubtatorAnnotation]) -> List[str]:
    """Converts the highly structed `pubtator_annotations` input to a format that can be used
    with seq2rel.
    """
    seq2rel_annotations = []

    for annotation in pubtator_annotations.values():
        relations = []
        offsets = []

        for rel in annotation.relations:
            uid_1, uid_2, rel_label = rel
            # Keep track of the end offsets of each entity. We will use these to sort
            # relations according to their order of first appearence in the text.
            offset_1 = min((end for _, end in annotation.clusters[uid_1].offsets))
            offset_2 = min((end for _, end in annotation.clusters[uid_2].offsets))
            offset = offset_1 + offset_2
            ent_clusters = [annotation.clusters[uid_1].ents, annotation.clusters[uid_2].ents]
            ent_labels = [annotation.clusters[uid_1].label, annotation.clusters[uid_2].label]
            relation = format_relation(
                ent_clusters=ent_clusters,
                ent_labels=ent_labels,
                rel_label=rel_label,
            )
            relations.append(relation)
            offsets.append(offset)

        relations = sort_by_offset(relations, offsets)
        seq2rel_annotations.append(f"{annotation.text}\t{' '.join(relations)}")

    return seq2rel_annotations


def compute_corpus_statistics(
    annotations: Dict[str, PubtatorAnnotation], nlp=None
) -> PartitionStatistics:
    num_relations = 0
    num_inter_sent = 0

    for doc_id, annotation in annotations.items():
        doc = nlp(annotation.text)
        num_relations += len(annotation.relations)
        for relation in annotation.relations:
            ent_1_offsets = annotation.clusters[relation[0]].offsets
            ent_2_offsets = annotation.clusters[relation[1]].offsets
            inter_sent = True
            for sent in doc.sents:
                # If a relation has two mentions within the same sentence,
                # consider it an intra-sentence relation.
                if any(
                    start >= sent.start_char and end <= sent.end_char
                    for start, end in ent_1_offsets
                ) and any(
                    start >= sent.start_char and end <= sent.end_char
                    for start, end in ent_2_offsets
                ):
                    inter_sent = False
                    break
            num_inter_sent += int(inter_sent)
    statistics = PartitionStatistics(
        num_examples=len(annotations),
        num_relations=num_relations,
        per_inter_sent=num_inter_sent / num_relations,
    )
    return statistics
