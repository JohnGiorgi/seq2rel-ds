import random

import numpy as np
import pytest

from seq2rel_ds.common import schemas, util


def test_find_first_mention() -> None:
    # string not in the text
    string = "Waldo"
    text = "He's not here!"
    match = util._find_first_mention(string, text)
    assert match is None
    # A false flag, Waldo appeares within another word before it appears on its own
    ent = "Waldo"
    text = "Waldorf is not Waldo"
    match = util._find_first_mention(ent, text)
    assert match.span() == (15, 20)
    # Check that we can match a compound entity lazily
    ent = "Waldo and Wally"
    text = "Waldo said to Wally, and Wally said to Waldo"
    match = util._find_first_mention(ent, text)
    assert match.span() == (0, 19)
    # Check that we can provide kwargs to Pattern.search
    ent = "Wally"
    text = "Waldo said to Wally, and Wally said to Waldo"
    match = util._find_first_mention(ent, text, pos=0)
    assert match.span() == (14, 19)
    match = util._find_first_mention(ent, text, pos=19)
    assert match.span() == (25, 30)


def test_set_seeds() -> None:
    # As it turns out, it is quite hard to determine the current random
    # seed. I was able to determine how for numpy, but not python or tensorflow.
    # See: https://stackoverflow.com/a/49749486/6578628
    util.set_seeds()
    assert np.random.get_state()[1][0] == util.NUMPY_SEED


def test_download_zip() -> None:
    # Download some dummy data and make sure we can read it.
    z = util.download_zip(
        "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip"
    )
    assert len(z.namelist()) == 1
    _ = z.read("sample.txt")


def test_train_valid_test_split():
    data = random.sample(range(0, 100), 10)
    train_size, valid_size, test_size = 0.7, 0.1, 0.2
    train, valid, test = util.train_valid_test_split(
        data, train_size=train_size, valid_size=valid_size, test_size=test_size
    )
    assert len(train) == int(train_size * len(data))
    assert len(valid) == int(valid_size * len(data))
    assert len(test) == int(test_size * len(data))


def test_train_valid_test_split_value_error():
    data = random.sample(range(0, 100), 10)
    # train_size + valid_size + test_size != 1
    train_size, valid_size, test_size = 0.7, 0.1, 0.3
    with pytest.raises(ValueError):
        _, _, _ = util.train_valid_test_split(
            data, train_size=train_size, valid_size=valid_size, test_size=test_size
        )


def test_parse_pubtator() -> None:
    # A truncated example taken from the CDR dataset
    pmid = "18020536"
    title_text = (
        "Associations between use of benzodiazepines or related drugs and health, physical"
        " abilities and cognitive function: a non-randomised clinical study in the elderly."
    )
    abstract_text = (
        "OBJECTIVE: To describe associations between the use of benzodiazepines or related drugs"
        " (BZDs/RDs) and health, functional abilities and cognitive function in the elderly."
        " METHODS: A non-randomised clinical study of patients aged > or =65 years admitted to"
        " acute hospital wards during 1 month. 164 patients (mean age +/- standard deviation [SD]"
        " 81.6 +/- 6.8 years) were admitted. Of these, nearly half (n = 78) had used BZDs/RDs"
        " before admission, and the remainder (n = 86) were non-users. Cognitive ability was"
        " assessed by the Mini-Mental State Examination (MMSE). Patients scoring > or =20 MMSE"
        " sum points were interviewed (n = 79) and questioned regarding symptoms and functional"
        " abilities during the week prior to admission."
    )
    # Include a dummy annotation with ID == -1. These should be ignored.
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}\t28\t43\tbenzodiazepines\tChemical\tD001569
    {pmid}\t219\t234\tbenzodiazepines\tChemical\tD001569
    {pmid}\t253\t257\tBZDs\tChemical\tD001569
    {pmid}\t583\t587\tBZDs\tChemical\tD001569
    {pmid}\t1817\t1826\ttiredness\tDisease\tD005221
    {pmid}\t0\t0\tArbitrary\tArbitrary\t-1
    {pmid}\tCID\tD001569\tD005221
    """

    title_entities = {
        "D001569": schemas.PubtatorEntity(
            mentions=["benzodiazepines"],
            offsets=[(28, 43)],
            label="Chemical",
        ),
    }
    abstract_entities = {
        "D001569": schemas.PubtatorEntity(
            mentions=["benzodiazepines", "BZDs", "BZDs"],
            offsets=[(219, 234), (253, 257), (583, 587)],
            label="Chemical",
        ),
        "D005221": schemas.PubtatorEntity(
            mentions=["tiredness"], offsets=[(1817, 1826)], label="Disease"
        ),
    }
    both_entities = {
        "D001569": schemas.PubtatorEntity(
            mentions=title_entities["D001569"].mentions + abstract_entities["D001569"].mentions,
            offsets=title_entities["D001569"].offsets + abstract_entities["D001569"].offsets,
            label="Chemical",
        ),
        "D005221": schemas.PubtatorEntity(
            mentions=["tiredness"], offsets=[(1817, 1826)], label="Disease"
        ),
    }

    # Title only
    expected = schemas.PubtatorAnnotation(
        pmid=pmid, text=title_text, entities=title_entities, relations=[]
    )
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.title)
    # Breaking up the asserts leads to much clearer outputs when the test fails
    assert actual[0].text == expected.text
    assert actual[0].entities == expected.entities
    assert actual[0].relations == expected.relations

    # Abstract only
    expected = schemas.PubtatorAnnotation(
        pmid=pmid,
        text=abstract_text,
        entities=abstract_entities,
        relations=[("D001569", "D005221", "CID")],
    )
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.abstract)
    assert actual[0].text == expected.text
    assert actual[0].entities == expected.entities
    assert actual[0].relations == expected.relations

    # Both
    expected = schemas.PubtatorAnnotation(
        pmid=pmid,
        text=f"{title_text} {abstract_text}",
        entities=both_entities,
        relations=[("D001569", "D005221", "CID")],
    )
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)
    assert actual[0].text == expected.text
    assert actual[0].entities == expected.entities
    assert actual[0].relations == expected.relations


def test_parse_pubtator_skip_malformed_raises_value_error() -> None:
    # A truncated example taken from the CDR dataset
    pmid = "2339463"
    title_text = "Cerebral sinus thrombosis as a potential hazard of antifibrinolytic treatment in menorrhagia."
    abstract_text = (
        "We describe a 42-year-old woman who developed superior sagittal and left transverse sinus"
        " thrombosis associated with prolonged epsilon-aminocaproic acid therapy for menorrhagia."
    )
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}\t0\t25\tCerebral sinus thrombosis\tDisease
    """

    # There should be no error when skip_malformed is True...
    _ = util.parse_pubtator(pubtator_content, skip_malformed=True)
    # ...and and error when it is False
    with pytest.raises(ValueError):
        _ = util.parse_pubtator(pubtator_content, skip_malformed=False)


def test_parse_pubtator_no_abstract_raises_value_error() -> None:
    # A truncated example taken from the CDR dataset
    pmid = "2339463"
    title_text = "Cerebral sinus thrombosis as a potential hazard of antifibrinolytic treatment in menorrhagia."
    abstract_text = ""
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}\t0\t25\tCerebral sinus thrombosis\tDisease
    """

    # A ValueError should be raised if no abstract provided but text_segment is "abstract" or "both"
    with pytest.raises(ValueError):
        _ = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.abstract)
    with pytest.raises(ValueError):
        _ = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)


def test_parse_pubtator_compound_ent() -> None:
    # A truncated example taken from the CDR dataset
    pmid = "17854040"
    title_text = (
        "Mutations associated with lamivudine-resistance in therapy-na  ve hepatitis B virus (HBV)"
        " infected patients with and without HIV co-infection: implications for antiretroviral"
        " therapy in HBV and HIV co-infected South African patients. infected patients with and"
        " without HIV co-infection: implications for antiretroviral therapy in HBV and HIV"
        " co-infected South African patients."
    )
    abstract_text = (
        "This was an exploratory study to investigate lamivudine-resistant hepatitis B virus (HBV)"
        " strains in selected lamivudine-na  ve HBV carriers with and without human"
        " immunodeficiency virus (HIV) co-infection in South African patients. Thirty-five"
        " lamivudine-naive HBV infected patients with or without HIV co-infection were studied: 15"
        " chronic HBV mono-infected patients and 20 HBV-HIV co-infected patients."
    )
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}\t26\t36\tlamivudine\tChemical\tD019259
    {pmid}\t59\t61\tna\tChemical\tD012964
    {pmid}\t66\t98\thepatitis B virus (HBV) infected\tDisease\tD006509
    {pmid}\t125\t141\tHIV co-infection\tDisease\tD015658
    {pmid}\t186\t209\tHBV and HIV co-infected\tDisease\tD006509|D015658	HBV infected|HIV infected
    """

    expected = schemas.PubtatorAnnotation(
        pmid=pmid,
        text=f"{title_text} {abstract_text}",
        entities={
            "D019259": schemas.PubtatorEntity(
                mentions=["lamivudine"],
                offsets=[(26, 36)],
                label="Chemical",
            ),
            "D012964": schemas.PubtatorEntity(
                mentions=["na"], offsets=[(59, 61)], label="Chemical"
            ),
            "D006509": schemas.PubtatorEntity(
                mentions=["hepatitis B virus (HBV) infected", "HBV infected"],
                offsets=[(66, 98), (186, 209)],
                label="Disease",
            ),
            "D015658": schemas.PubtatorEntity(
                mentions=["HIV co-infection", "HIV infected"],
                offsets=[(125, 141), (194, 209)],
                label="Disease",
            ),
        },
    )
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)
    assert actual[0].text == expected.text
    assert actual[0].entities == expected.entities
    assert actual[0].relations == expected.relations


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
    title_entities = {
        "6048": schemas.PubtatorEntity(
            mentions=["RNF5"],
            offsets=[(21, 25)],
            label="Gene",
        ),
        "340061": schemas.PubtatorEntity(mentions=["MITA"], offsets=[(104, 108)], label="Gene"),
    }
    abstract_entities = {
        "4790": schemas.PubtatorEntity(mentions=["NF-kappaB"], offsets=[(158, 167)], label="Gene"),
        "3661": schemas.PubtatorEntity(
            mentions=["IRF3", "IRF3", "IRF3"],
            offsets=[(172, 176), (378, 382), (554, 558)],
            label="Gene",
        ),
        "340061": schemas.PubtatorEntity(
            mentions=["MITA", "STING", "MITA", "MITA", "MITA", "MITA", "MITA"],
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
        "6048": schemas.PubtatorEntity(
            mentions=["RNF5", "RNF5", "RNF5", "RNF5", "RNF5", "RNF5", "RNF5"],
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
        "3456": schemas.PubtatorEntity(mentions=["IFNB1"], offsets=[(571, 576)], label="Gene"),
    }
    both_entities = {
        "6048": schemas.PubtatorEntity(
            mentions=title_entities["6048"].mentions + abstract_entities["6048"].mentions,
            offsets=title_entities["6048"].offsets + abstract_entities["6048"].offsets,
            label="Gene",
        ),
        "340061": schemas.PubtatorEntity(
            mentions=title_entities["340061"].mentions + abstract_entities["340061"].mentions,
            offsets=title_entities["340061"].offsets + abstract_entities["340061"].offsets,
            label="Gene",
        ),
        "4790": schemas.PubtatorEntity(mentions=["NF-kappaB"], offsets=[(158, 167)], label="Gene"),
        "3661": schemas.PubtatorEntity(
            mentions=abstract_entities["3661"].mentions,
            offsets=abstract_entities["3661"].offsets,
            label="Gene",
        ),
        "3456": schemas.PubtatorEntity(mentions=["IFNB1"], offsets=[(571, 576)], label="Gene"),
    }

    # Title only
    expected = schemas.PubtatorAnnotation(
        pmid=pmid, text=title_text, entities=title_entities, relations=[]
    )
    actual = util.query_pubtator(
        pmids=[pmid], concepts=["gene"], text_segment=util.TextSegment.title
    )
    # Breaking up the asserts leads to much clearer outputs when the test fails
    assert len(actual) == 1
    assert actual[expected.pmid].text == expected.text
    assert actual[expected.pmid].entities == expected.entities
    assert actual[expected.pmid].relations == expected.relations

    # Abstract only
    expected = schemas.PubtatorAnnotation(
        pmid=pmid, text=abstract_text, entities=abstract_entities, relations=[]
    )
    actual = util.query_pubtator(
        pmids=[pmid], concepts=["gene"], text_segment=util.TextSegment.abstract
    )
    assert len(actual) == 1
    assert actual[expected.pmid].text == expected.text
    assert actual[expected.pmid].entities == expected.entities
    assert actual[expected.pmid].relations == expected.relations

    # Both
    expected = schemas.PubtatorAnnotation(
        pmid=pmid, text=f"{title_text} {abstract_text}", entities=both_entities, relations=[]
    )
    actual = util.query_pubtator(
        pmids=[pmid], concepts=["gene"], text_segment=util.TextSegment.both
    )
    assert len(actual) == 1
    assert actual[expected.pmid].text == expected.text
    assert actual[expected.pmid].entities == expected.entities
    assert actual[expected.pmid].relations == expected.relations
