from seq2rel_ds.align import util
from seq2rel_ds.common import schemas
from seq2rel_ds.common.util import TextSegment


def test_query_pubtator() -> None:
    pmid = "19285439"
    title_text = (
        "The ubiquitin ligase RNF5 regulates antiviral responses by mediating degradation"
        " of the adaptor protein MITA."
    )
    abstract_text = (
        "Viral infection activates transcription factors NF-kappaB and IRF3, which collaborate to"
        " induce type I interferons (IFNs) and elicit innate antiviral response. MITA (also known"
        " as STING) has recently been identified as an adaptor that links virus-sensing receptors"
        " to IRF3 activation. Here, we showed that the E3 ubiquitin ligase RNF5 interacted with"
        " MITA in a viral-infection-dependent manner. Overexpression of RNF5 inhibited"
        " virus-triggered IRF3 activation, IFNB1 expression, and cellular antiviral response,"
        " whereas knockdown of RNF5 had opposite effects. RNF5 targeted MITA at Lys150 for"
        " ubiquitination and degradation after viral infection. Both MITA and RNF5 were located at"
        " the mitochondria and endoplasmic reticulum (ER) and viral infection caused their"
        " redistribution to the ER and mitochondria, respectively. We further found that"
        " virus-induced ubiquitination and degradation of MITA by RNF5 occurred at the"
        " mitochondria. These findings suggest that RNF5 negatively regulates virus-triggered"
        " signaling by targeting MITA for ubiquitination and degradation at the mitochondria."
    )
    title_clusters = {
        "6048": schemas.PubtatorCluster(
            ents=["rnf5"],
            offsets=[(21, 25)],
            label="Gene",
        ),
        "340061": schemas.PubtatorCluster(ents=["mita"], offsets=[(104, 108)], label="Gene"),
    }
    abstract_clusters = {
        "4790": schemas.PubtatorCluster(ents=["nf-kappab"], offsets=[(158, 167)], label="Gene"),
        "3661": schemas.PubtatorCluster(
            ents=["irf3"], offsets=[(172, 176), (378, 382), (554, 558)], label="Gene"
        ),
        "340061": schemas.PubtatorCluster(
            ents=["mita", "sting"],
            offsets=[
                (270, 274),
                (290, 295),
                (461, 465),
                (684, 688),
                (762, 766),
                (1000, 1004),
                (1136, 1140),
            ],
            label="Gene",
        ),
        "6048": schemas.PubtatorCluster(
            ents=["rnf5"],
            offsets=[
                (440, 444),
                (523, 527),
                (643, 647),
                (670, 674),
                (771, 775),
                (1008, 1012),
                (1071, 1075),
            ],
            label="Gene",
        ),
        "3456": schemas.PubtatorCluster(ents=["ifnb1"], offsets=[(571, 576)], label="Gene"),
    }
    both_clusters = {
        "6048": schemas.PubtatorCluster(
            ents=["rnf5"],
            offsets=[
                (21, 25),
                (440, 444),
                (523, 527),
                (643, 647),
                (670, 674),
                (771, 775),
                (1008, 1012),
                (1071, 1075),
            ],
            label="Gene",
        ),
        "340061": schemas.PubtatorCluster(
            ents=["mita", "sting"],
            offsets=[
                (104, 108),
                (270, 274),
                (290, 295),
                (461, 465),
                (684, 688),
                (762, 766),
                (1000, 1004),
                (1136, 1140),
            ],
            label="Gene",
        ),
        "4790": schemas.PubtatorCluster(ents=["nf-kappab"], offsets=[(158, 167)], label="Gene"),
        "3661": schemas.PubtatorCluster(
            ents=["irf3"], offsets=[(172, 176), (378, 382), (554, 558)], label="Gene"
        ),
        "3456": schemas.PubtatorCluster(ents=["ifnb1"], offsets=[(571, 576)], label="Gene"),
    }

    # Title only
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=title_text,
            clusters=title_clusters,
            relations=[],
        ),
    }
    actual = util.query_pubtator(pmids=[pmid], concepts=["gene"], text_segment=TextSegment.title)
    # Breaking up the asserts leads to much clearer outputs when the test fails
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations

    # Abstract only
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=abstract_text,
            clusters=abstract_clusters,
            relations=[],
        ),
    }
    actual = util.query_pubtator(pmids=[pmid], concepts=["gene"], text_segment=TextSegment.abstract)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations

    # Both
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=f"{title_text} {abstract_text}",
            clusters=both_clusters,
            relations=[],
        ),
    }
    actual = util.query_pubtator(pmids=[pmid], concepts=["gene"], text_segment=TextSegment.both)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations
