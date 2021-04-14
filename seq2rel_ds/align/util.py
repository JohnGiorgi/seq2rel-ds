import re
from io import StringIO
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from seq2rel_ds.common.util import sanitize_text
from urllib3.util.retry import Retry

UNIPROT_API_URL = "https://www.uniprot.org/uploadlists/"
PUBTATOR_API_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"

PROTEIN_NAME_REGEX = r"(?:(?:Full|Short)=)(?!\s)([^{;]*)(?<!\s)"
GENE_NAME_REGEX = r"(?:Name=)(?!\s)([^{;]*)(?<!\s)"
GENE_SYNONYMS_REGEX = r"(?:Synonyms=)(?!\s)([^{;]*)(?<!\s)"

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

# Some custom types
Synonyms = Dict[str, List[str]]
Annotations = Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]]


def get_uniprot_synonyms(uniprot_ids: List[str]) -> Synonyms:
    """Returns a dictionary, keyed by the ids in `uniprot_ids`, containing each ids gene and
    protein name synonyms.
    """
    synonyms = {}
    # See UniProts REST API for details: https://www.uniprot.org/help/api
    file = StringIO(" ".join(uniprot_ids))
    params = {"format": "txt", "from": "ACC+ID", "to": "ACC"}
    r = s.post(UNIPROT_API_URL, files={"file": file}, params=params)
    entries = r.text.strip().split("\n//")

    for uniprot_id, entry in zip(uniprot_ids, entries):
        synonyms[uniprot_id] = []
        for line in entry.split("\n"):
            if line.startswith("DE"):
                synonyms[uniprot_id].extend(re.findall(PROTEIN_NAME_REGEX, line))
            elif line.startswith("GN"):
                synonyms[uniprot_id].extend(re.findall(GENE_NAME_REGEX, line))
                gene_name_syns = re.findall(GENE_SYNONYMS_REGEX, line)
                if gene_name_syns:
                    synonyms[uniprot_id].extend(gene_name_syns[0].split(", "))
    return synonyms


def get_pubtator_response(pmids: List[str], concepts: Optional[List[str]] = None) -> Annotations:
    json_body = {"pmids": pmids}
    if concepts is not None:
        json_body["concepts"] = concepts
    r = s.post(PUBTATOR_API_URL, json=json_body)
    articles = r.text.strip().split("\n\n")
    annotations = {}
    for article in articles:
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
