import io
import random
import re
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import requests
import spacy
from more_itertools import chunked
from requests.adapters import HTTPAdapter
from sklearn.model_selection import train_test_split
from urllib3.util.retry import Retry

from seq2rel_ds.common import sorting_utils
from seq2rel_ds.common.schemas import PartitionStatistics, PubtatorAnnotation, PubtatorCluster

# Seeds
SEED = 13370
NUMPY_SEED = 1337

# API URLs
PUBTATOR_API_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"


# Enums
class TextSegment(str, Enum):
    title = "title"  # type: ignore
    abstract = "abstract"
    both = "both"


class EntityHinting(str, Enum):
    gold = "gold"
    pipeline = "pipeline"


# Here, we create a session globally that can be used in all requests. We add a hook that will
# call raise_for_status() after all our requests. Because API calls can be flaky, we also add
# multiple requests with backoff.
# Details here: https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
# and here: https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
s = requests.Session()
assert_status_hook = lambda response, *args, **kwargs: response.raise_for_status()  # noqa
s.hooks["response"] = [assert_status_hook]
retries = Retry(total=5, backoff_factor=0.1)
s.mount("https://", HTTPAdapter(max_retries=retries))


# Private functions #


def _find_first_mention(string: str, text: str, **kwargs: Any) -> Optional[re.Match]:
    """Search for the first occurrence of `string` in `text`, returning an `re.Match` object if
    found and `None` otherwise. To match `string` to `text` more accurately, we use a type of
    "backoff" strategy. First, we look for the whole entity in text. If we cannot find it, we look
    for a lazy match of its first and last tokens. `**kwargs` are passed to `Pattern.search`.
    """
    match = re.compile(rf"\b{re.escape(string)}\b").search(text, **kwargs)

    if not match:
        ent_split = string.split()
        if len(ent_split) > 1:
            first, last = re.escape(ent_split[0]), re.escape(ent_split[-1])
            match = re.compile(rf"\b{first}.*?{last}\b").search(text, **kwargs)
    return match


def _query_pubtator(body: Dict[str, Any], **kwargs: Any) -> Dict[str, PubtatorAnnotation]:
    r = s.post(PUBTATOR_API_URL, json=body)
    pubtator_content = r.text.strip()
    pubtator_annotations = parse_pubtator(pubtator_content, **kwargs)
    return {ann.pmid: ann for ann in pubtator_annotations}


# Public functions #


def set_seeds() -> None:
    """Sets the random seeds of python and numpy for reproducible preprocessing."""
    random.seed(SEED)
    np.random.seed(NUMPY_SEED)


def download_zip(url: str) -> ZipFile:
    # https://stackoverflow.com/a/23419450/6578628
    r = requests.get(url)
    z = ZipFile(io.BytesIO(r.content))
    return z


def train_valid_test_split(
    data: Iterable[Any],
    train_size: float = 0.7,
    valid_size: float = 0.1,
    test_size: float = 0.2,
    **kwargs: Any,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Given an iterable (`data`), returns train, valid and test partitions of size `train_size`,
    `valid_size` and `test_size`. Optional `**kwargs` are passed to `sklearn.model_selection.train_test_split`.

    See https://datascience.stackexchange.com/a/53161 for details.
    """
    size_sum = train_size + valid_size + test_size
    if size_sum != 1.0:
        raise ValueError(f"train_size, valid_size and test_size must sum to one. Got {size_sum}.")
    # Round to avoid precision errors.
    train, test = train_test_split(data, test_size=round(1 - train_size, 4), **kwargs)
    valid, test = train_test_split(test, test_size=test_size / (test_size + valid_size), **kwargs)
    return train, valid, test


def compute_corpus_statistics(
    annotations: List[PubtatorAnnotation],
    nlp=None,
) -> PartitionStatistics:
    """
    Computes several corpus statistics give a list of `PubtatorAnnotation` objects, including
    number of examples, number of relations, and the fraction of relations that are inter-sentence.
    """
    num_relations = 0
    num_inter_sent = 0

    # Not ideal to load the model every time the function is called, but it is fast enough.
    nlp = nlp or spacy.load("en_core_web_sm", disable=["ner"])

    # Faster to process all text with `pipe` than to process text one at a time.
    # See: https://spacy.io/usage/processing-pipelines#processing
    docs = list(nlp.pipe([ann.text for ann in annotations]))

    for ann, doc in zip(annotations, docs):
        print(ann)
        print()
        num_relations += len(ann.relations)
        for relation in ann.relations:
            inter_sent = True
            # Collect the offsets of all mentions off all entities in the relation.
            mention_offsets = [ann.clusters[relation[i]].offsets for i in range(len(relation) - 1)]
            # If any sentence in the document contains at least one mention from each entity in a
            # relation, we consider it an intra-sentence relation.
            for sent in doc.sents:
                if all(
                    all(start >= sent.start_char and end <= sent.end_char for start, end in offsets)
                    for offsets in mention_offsets
                ):
                    inter_sent = False
                    break
            num_inter_sent += int(inter_sent)

    statistics = PartitionStatistics(
        num_examples=len(annotations),
        num_relations=num_relations,
        per_inter_sent=round(num_inter_sent / num_relations, 3),
    )
    return statistics


def parse_pubtator(
    pubtator_content: str,
    text_segment: TextSegment = TextSegment.both,
    skip_malformed: bool = False,
) -> List[PubtatorAnnotation]:
    """Parses a PubTator formatted string (`pubtator_content`) and returns a list of
    `PubtatorAnnotation` objects.

    # Parameters

    pubtator_content : `str`
        A string containing one or more articles in the PubTator format.
    text_segment : `TextSegment`, optional (default = `TextSegment.both`)
        Which segment of the text we should consider. Valid values are `TextSegment.title`,
        `TextSegment.abstract` or `TextSegment.both`.
    skip_malformed : bool, optional (default = `False`)
        True if we should ignore malformed annotations that cannot be parsed. This is useful in
        some cases, like when we are generating data using distant supervision.
    """
    # Get a list of PubTator annotations
    articles = pubtator_content.strip().split("\n\n")

    # Parse the annotations, producing a highly structured output
    parsed = []
    for article in articles:
        # Extract the title and abstract (if it exists)
        split_article = article.strip().split("\n")
        title, abstract, annotations = split_article[0], split_article[1], split_article[2:]
        pmid, title = title.split("|t|")
        abstract = abstract.split("|a|")[-1]
        title = title.strip()
        abstract = abstract.strip()

        # Sort mentions by order of first appearance
        annotations = sorting_utils.sort_entity_annotations(annotations)

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

        parsed.append(PubtatorAnnotation(pmid=pmid, text=text))

        for ann in annotations:
            split_ann = ann.strip().split("\t")

            # This is a entity mention
            start: int
            end: int
            if sorting_utils.pubtator_ann_is_mention(split_ann):
                if len(split_ann) == 6:
                    _, start, end, mentions, label, uids = split_ann  # type: ignore
                elif len(split_ann) == 7:
                    _, start, end, _, label, uids, mentions = split_ann  # type: ignore
                # For some cases (like distant supervision) it is
                # convenient to skip annotations that are malformed.
                else:
                    if skip_malformed:
                        continue
                    else:
                        err_msg = f"Found an annotation with an unexpected number of columns: {ann}"
                        raise ValueError(err_msg)
                start, end = int(start), int(end)

                # Ignore this annotation if it is not in the chosen text segment
                section = "title" if start < len(title) else "abstract"
                if section != text_segment.value and text_segment.value != "both":
                    continue

                # In at least one corpus (CDR), the annotators include annotations for the
                # individual entities in a compound entity. So we deal with that here.
                # Note that the start & end indicies will no longer be exactly correct, but are
                # be close enough for our purposes of sorting entities by order of appearence.
                mentions, uids = mentions.split("|"), uids.split("|")  # type: ignore
                for mention, uid in zip(mentions, uids):
                    # Ignore this annotation if the entity is not grounded.
                    # à la: https://www.aclweb.org/anthology/D19-1498/
                    if uid == "-1":
                        continue

                    offset = (start, end)

                    # If this is a compound entity update the offsets to be as correct as possible.
                    if len(mentions) > 1:
                        match = _find_first_mention(mention, text, pos=start, endpos=end)
                        if match is not None:
                            offset = match.span()

                    if uid in parsed[-1].clusters:
                        parsed[-1].clusters[uid].mentions.append(mention)
                        parsed[-1].clusters[uid].offsets.append(offset)
                    else:
                        parsed[-1].clusters[uid] = PubtatorCluster(
                            mentions=[mention], offsets=[offset], label=label
                        )
            # This is a relation
            else:
                _, label, *uids = split_ann  # type: ignore
                rel = (*uids, label)
                # Check that the relations entities are in the text
                # and that this relation is unique.
                if rel not in parsed[-1].relations and all(
                    uid in parsed[-1].clusters for uid in uids
                ):
                    parsed[-1].relations.append(rel)

    return parsed


def pubtator_to_seq2rel(
    document_annotations: List[PubtatorAnnotation],
    sort_rels: bool = True,
    entity_hinting: Optional[EntityHinting] = None,
    **kwargs: Any,
) -> List[str]:
    """Converts the highly structured `pubtator_annotations` input to a format that can be used with
    seq2rel. Optional `**kwargs` are passed to `query_pubtator` when `entity_hinting == "pipeline"`.

    # Parameters

    document_annotations : `List[PubtatorAnnotation]`
        A list of`PubtatorAnnotation` objects to convert to the seq2rel format.
    sort_rels : bool, optional (default = `True`)
        Whether relations should be sorted by order of first appearance. This useful for traditional
        seq2seq models that use an order-sensitive loss function, like negative log-likelihood.
    include_ent_hints : bool, optional (default = `False`)
        True if entity markers should be included within the source text. This effectively converts
        the end-to-end relation extraction problem to a pipeline relation extraction approach, where
        entities are given.
    """
    seq2rel_annotations = []

    # If using pipeline-based entity hinting, it is much faster to retrieve the annotations in bulk
    pmids = [ann.pmid for ann in document_annotations]
    pubtator_annotations = (
        query_pubtator(pmids, **kwargs) if entity_hinting == EntityHinting.pipeline else {}
    )

    for doc_ann in document_annotations:
        # Apply entity hinting using the requested strategy (if any). In the "pipeline" setting
        # we use the annotations from PubTator to determine the entity hints. Otherwise, we use
        # the ground truth annotations.
        if entity_hinting == EntityHinting.pipeline:
            pubtator_ann = pubtator_annotations.get(doc_ann.pmid)
            if pubtator_ann is None:
                warnings.warn(
                    f"{entity_hinting} entity hinting strategy selected, but no annotations found"
                    f" for PMID: {doc_ann.pmid}. No hints will be inserted for this document."
                )
                continue
            pubtator_ann.insert_hints()
            doc_ann.text = pubtator_ann.text
        elif entity_hinting == EntityHinting.gold:
            doc_ann.insert_hints()

        relation_string = doc_ann.to_string(sort=sort_rels)
        seq2rel_annotation = f"{doc_ann.text.strip()}\t{relation_string.strip()}"
        seq2rel_annotations.append(seq2rel_annotation)

    return seq2rel_annotations


def query_pubtator(
    pmids: List[str],
    concepts: Optional[List[str]] = None,
    pmids_per_request: int = 1000,
    **kwargs: Any,
) -> Dict[str, PubtatorAnnotation]:
    """Queries PubTator for the given `pmids` and `concepts`, parses the results and
    returns a highly structured dictionary-like object keyed by PMID. Optional `**kwargs` are passed
    to `seq2rel_ds.common.util.parse_pubtator`.
    For details on the PubTator API, see: https://www.ncbi.nlm.nih.gov/research/pubtator/api.html

    # Parameters

    pmids : `List[str]`
        A list of PMIDs to query PubTator with.
    concepts : `List[str]`, optional (default = `None`)
        A list of concepts to include in the PubTator results.

    """
    body: Dict[str, Any] = {"type": "pmids"}
    if concepts is not None:
        body["concepts"] = concepts
    annotations = {}
    for chunk in chunked(pmids, pmids_per_request):
        # Try to post requests in chunks to speed things up...
        try:
            body["pmids"] = chunk
            pubtator_annotations = _query_pubtator(body, **kwargs)
            annotations.update(pubtator_annotations)
        # ...but, if the request fails, recursively half the size of the request.
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            pubtator_annotations = query_pubtator(chunk, concepts, pmids_per_request // 2, **kwargs)
            annotations.update(pubtator_annotations)
    return annotations
