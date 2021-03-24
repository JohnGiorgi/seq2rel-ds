import random
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from xml.parsers.expat import ExpatError

import numpy as np
import requests
import xmltodict
from fuzzywuzzy import process
from joblib import Memory
from sklearn.model_selection import train_test_split

SEED = 13370
NUMPY_SEED = 1337

# TODO: Would be better to move this to an __init__ so all files in the directory can use it.
# Setup caching
CACHE_DIR = Path.home() / ".seq2rel_ds"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)
memory = Memory(CACHE_DIR, verbose=0)


def set_seeds():
    """Sets the random seeds of python and numpy for reproducible preprocessing."""
    random.seed(SEED)
    np.random.seed(NUMPY_SEED)


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


@memory.cache()
def get_uniprot_synonyms(uniprot_id: str) -> List[str]:
    """Returns all full and short names from UniProt for a protein with ID `uniprot_id`."""
    synonyms = []
    # TODO: We should be using the batch request here.
    # See UniProts REST API for details: https://www.uniprot.org/help/api\
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    r = requests.get(url)
    r.raise_for_status()
    # Do some basic XML parsing
    try:
        parsed_response = xmltodict.parse(r.text)
    except ExpatError:
        return synonyms
    try:
        protein = parsed_response["uniprot"]["entry"]["protein"]
    except KeyError:
        return synonyms
    recommended_name = protein.get("recommendedName", [])
    alternative_name = protein.get("alternativeName", [])
    if not isinstance(recommended_name, list):
        recommended_name = [recommended_name]
    if not isinstance(alternative_name, list):
        alternative_name = [alternative_name]
    names = recommended_name + alternative_name
    # Retrive a list of long and short name synonyms
    for name in names:
        full_name = name.get("fullName")
        short_name = name.get("shortName")
        if full_name:
            if not isinstance(full_name, list):
                full_name = [full_name]
            # May have attributes, so extract the text
            full_name = [fn if isinstance(fn, str) else fn.get("#text", fn) for fn in full_name]
            synonyms.extend(full_name)
        if short_name:
            if not isinstance(short_name, list):
                short_name = [short_name]
            short_name = [sn if isinstance(sn, str) else sn.get("#text", sn) for sn in short_name]
            synonyms.extend(short_name)

    return synonyms


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


def get_pubtator_response(
    pmids: List[str], concepts: Optional[List[str]] = None
) -> Tuple[str, str, List[str]]:
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"
    json = {"pmids": pmids}
    if concepts is not None:
        json["concepts"] = concepts
    response = requests.post(url, json=json)
    response = response.text.strip().split("\n\n")
    annotations = {}
    for article in response:
        article = article.strip().split("\n")
        title, abstract, ents = article[0], article[1], article[2:]
        pmid, title = title.split("|t|")
        abstract = abstract.split("|a|")[-1]

        # Some very basic preprocessing
        title = sanitize_text(title)
        abstract = sanitize_text(abstract)

        title_ents, abstract_ents = [], []
        for ent in ents:
            ent = ent.strip().split("\t")
            start, text = ent[1], ent[3]
            if int(start) < len(title):
                title_ents.append(text)
            else:
                abstract_ents.append(text)

        annotations[pmid] = {
            "title": {"text": title, "ents": title_ents},
            "abstract": {"text": abstract, "ents": abstract_ents},
        }
    return annotations
