from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from seq2rel_ds.common import util
from seq2rel_ds.common.schemas import PubtatorAnnotation
from seq2rel_ds.common.util import TextSegment
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


def query_pubtator(
    pmids: List[str],
    concepts: Optional[List[str]] = None,
    text_segment: TextSegment = TextSegment.both,
    skip_malformed: bool = False,
) -> Dict[str, PubtatorAnnotation]:
    body = {"pmids": pmids}
    if concepts is not None:
        body["concepts"] = concepts
    r = s.post(PUBTATOR_API_URL, json=body)
    pubtator_content = r.text.strip()
    annotations = util.parse_pubtator(
        pubtator_content, text_segment=text_segment, skip_malformed=skip_malformed
    )
    return annotations
