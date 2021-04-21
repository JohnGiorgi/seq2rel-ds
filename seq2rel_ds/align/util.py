from typing import Optional, List

import requests
from requests.adapters import HTTPAdapter
from seq2rel_ds.common.util import sanitize_text
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


def get_pubtator_response(pmids: List[str], concepts: Optional[List[str]] = None):
    body = {"pmids": pmids}
    if concepts is not None:
        body["concepts"] = concepts
    r = s.post(PUBTATOR_API_URL, json=body)
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

        annotations[pmid] = {
            "title": {"text": title, "clusters": {}},
            "abstract": {"text": abstract, "clusters": {}},
        }

        for ent in ents:
            ent_data = ent.strip().split("\t")
            # A very small number of ents are missing IDs. Skip them.
            if len(ent_data) < 6:
                continue
            _, start, end, text, _, entrezgene = ent_data
            text = text.lower()
            start, end = int(start), int(end)

            # Collect entities per section
            section = "title" if int(start) < len(title) else "abstract"

            if entrezgene in annotations[pmid][section]["clusters"]:
                # Don't retain duplicates
                if text not in annotations[pmid][section]["clusters"][entrezgene]["ents"]:
                    annotations[pmid][section]["clusters"][entrezgene]["ents"].append(text)
                    annotations[pmid][section]["clusters"][entrezgene]["offsets"].append(
                        (start, end)
                    )
            else:
                annotations[pmid][section]["clusters"][entrezgene] = {
                    "ents": [text],
                    "offsets": [(start, end)],
                }

    return annotations
