from typing import Any, Dict, List, Optional

import requests
from more_itertools import chunked
from requests.adapters import HTTPAdapter
from seq2rel_ds.common import util
from seq2rel_ds.common.schemas import PubtatorAnnotation
from urllib3.util.retry import Retry

# API URLs
PUBTATOR_API_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"

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


def _query_pubtator(body: Dict[str, Any], **kwargs):
    r = s.post(PUBTATOR_API_URL, json=body)
    pubtator_content = r.text.strip()
    pubtator_annotations = util.parse_pubtator(pubtator_content, **kwargs)
    return pubtator_annotations


def query_pubtator(
    pmids: List[str], concepts: Optional[List[str]] = None, pmids_per_request: int = 1000, **kwargs
) -> Dict[str, PubtatorAnnotation]:
    """Queries PubTator for the given `pmids` and `concepts`, parses the results and
    returns a highly structured dictionary-like object keyed by PMID. `**kwargs` are passed to
    `seq2rel_ds.common.util.parse_pubtator`.
    For details on the PubTator API, see: https://www.ncbi.nlm.nih.gov/research/pubtator/api.html

    # Parameters

    pmids : `List[str]`
        A list of PMIDs to query PubTator with.
    concepts : `List[str]`, optional (default = `None`)
        A list of concepts to include in the PubTator results.
    """
    body = {}
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
        except requests.exceptions.ConnectionError:
            pubtator_annotations = query_pubtator(chunk, concepts, pmids_per_request // 2, **kwargs)
            annotations.update(pubtator_annotations)
    return annotations
