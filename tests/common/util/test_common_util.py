import copy
import random
import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import booleans, text
from seq2rel_ds.common import schemas, util

# Private functions #


def test_search_ent() -> None:
    # Entity not in the sentence
    ent = "Waldo"
    text = "He's not here!"
    match = util._search_ent(ent, text)
    assert match is None
    # A false flag, Waldo appeares within another word before it appears on its own
    ent = "Waldo"
    text = "Waldorf is not Waldo"
    match = util._search_ent(ent, text)
    assert match.span() == (15, 20)
    # Check that we can match a compound entity lazily
    ent = "Waldo and Wally"
    text = "Waldo said to Wally, and Wally said to Waldo"
    match = util._search_ent(ent, text)
    assert match.span() == (0, 19)


def test_insert_ent_hints() -> None:
    """Asserts that insert_ent_hints works as expected for a list of edge cases."""
    # A truncated example taken from the GDA dataset. It contains a few edge cases:
    # - coreferent mention
    # - entites that differ in case
    # - paranthesized entity
    # - multiple identical mentions of an entity
    text = (
        "Apolipoprotein E epsilon4 allele, elevated midlife total cholesterol level, and high"
        " midlife systolic blood pressure are independent risk factors for late-life Alzheimer disease."
        # A + was added to see if the method can handle regex characters in the text
        " BACKGROUND: Presence of the apolipoprotein E (apoE+) epsilon4 allele, which is involved in"
        " cholesterol metabolism, is the most important genetic risk factor for Alzheimer disease."
        " Elevated midlife values for total cholesterol level and blood pressure have been"
        " implicated recently as risk factors for Alzheimer disease."
    )

    pubator_annotation = schemas.PubtatorAnnotation(
        text=text,
        clusters={
            # These are out of order on purpose, to ensure the function is insensitive to it
            "348": schemas.PubtatorCluster(
                ents=["Apolipoprotein E", "apoE+"], offsets=[(225, 229), (0, 17)], label="Gene"
            ),
            "D000544": schemas.PubtatorCluster(
                ents=["Alzheimer disease"],
                offsets=[(160, 177), (339, 356), (479, 496)],
                label="Disease",
            ),
        },
    )
    expected = copy.deepcopy(pubator_annotation)
    actual = util._insert_ent_hints(pubator_annotation)
    expected.text = (
        "@START_GENE@ Apolipoprotein E ; 0 @END_GENE@ epsilon4 allele, elevated midlife total cholesterol"
        " level, and high midlife systolic blood pressure are independent risk factors for late-life"
        " @START_DISEASE@ Alzheimer disease @END_DISEASE@ . BACKGROUND: Presence of the"
        " apolipoprotein E ( @START_GENE@ apoE+ ; 0 @END_GENE@ ) epsilon4 allele,"
        " which is involved in cholesterol metabolism, is the most important genetic risk factor for"
        " Alzheimer disease. Elevated midlife values for total cholesterol level and blood pressure"
        " have been implicated recently as risk factors for Alzheimer disease."
    )

    assert actual.text == expected.text
    assert actual.clusters == expected.clusters
    assert actual.relations == expected.relations


def test_insert_ent_hints_compound() -> None:
    """Asserts that insert_ent_hints works as expected for compound entities."""
    text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " intrahepatic and extrahepatic cholestasis."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        text=text,
        clusters={
            "D002780": schemas.PubtatorCluster(
                ents=["intrahepatic cholestasis"], offsets=[(87, 128)], label="Disease"
            ),
            "D001651": schemas.PubtatorCluster(
                ents=["extrahepatic cholestasis"], offsets=[(104, 128)], label="Disease"
            ),
        },
    )
    expected = copy.deepcopy(pubator_annotation)
    actual = util._insert_ent_hints(pubator_annotation)
    expected.text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " @START_DISEASE@ intrahepatic and @START_DISEASE@ extrahepatic cholestasis @END_DISEASE@ @END_DISEASE@ ."
    )
    assert actual.text == expected.text


def test_insert_ent_hints_overlapping() -> None:
    """Asserts that insert_ent_hints works as expected for overlapping entities."""
    text = (
        # A + was added to see if the method can handle regex characters in the text
        "Mutation pattern in clinically asymptomatic coagulation factor VII+ deficiency. A total of"
        " 122 subjects, referred after presurgery screening or checkup for prolonged prothrombin"
        " time, were characterized for the presence of coagulation factor VII deficiency."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        text=text,
        clusters={
            "2155": schemas.PubtatorCluster(
                ents=["coagulation factor VII+"], offsets=[(44, 67)], label="Gene"
            ),
            "D005168": schemas.PubtatorCluster(
                ents=["factor VII+ deficiency"], offsets=[(56, 78)], label="Disease"
            ),
        },
    )
    expected = copy.deepcopy(pubator_annotation)
    actual = util._insert_ent_hints(pubator_annotation)
    expected.text = (
        "Mutation pattern in clinically asymptomatic"
        " @START_GENE@ coagulation @START_DISEASE@ factor VII+ @END_GENE@ deficiency @END_DISEASE@ ."
        " A total of 122 subjects, referred after presurgery screening or checkup for prolonged"
        " prothrombin time, were characterized for the presence of coagulation factor VII deficiency."
    )
    assert actual.text == expected.text
    assert actual.clusters == expected.clusters
    assert actual.relations == expected.relations


# Public functions #


def test_set_seeds() -> None:
    # As it turns out, it is quite hard to determine the current random
    # seed. I was able to determine how for numpy, but not python or tensorflow.
    # See: https://stackoverflow.com/a/49749486/6578628
    util.set_seeds()
    assert np.random.get_state()[1][0] == util.NUMPY_SEED


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized_text = util.sanitize_text(text, lowercase=lowercase)

    # There should be no cases of multiple spaces or tabs
    assert re.search(r"[ ]{2,}", sanitized_text) is None
    assert "\t" not in sanitized_text
    # The beginning and end of the string should be stripped of whitespace
    assert not sanitized_text.startswith(("\n", " "))
    assert not sanitized_text.endswith(("\n", " "))
    # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
    # We don't particularly care about this, and it breaks this check.
    # Only run if the generated text can be lowercased.
    if lowercase and text.lower().islower():
        assert all(not char.isupper() for char in sanitized_text)


def test_sort_by_offset() -> None:
    items = ["b", "c", "a"]
    offsets = [1, 2, 0]

    expected = ["a", "b", "c"]
    actual = util.sort_by_offset(items, offsets)

    assert actual == expected
    # Check that we did not mutate the input
    assert items == ["b", "c", "a"]


def test_sort_by_offset_raise_value_error() -> None:
    items = ["b", "c", "a"]
    offsets = [1, 2]

    with pytest.raises(ValueError):
        _ = util.sort_by_offset(items, offsets)


def test_format_relation() -> None:
    # Add trailing and leading spaces throughout to ensure they are handled.
    rel_label = "Interaction "
    ent_clusters = [["MITA ", "STING"], [" NF-kappaB"], ["IRF3"]]
    ent_labels = ["GGP", " GGP", "GGP "]
    expected = (
        f"@INTERACTION@ mita ; sting @GGP@ nf-kappab @GGP@ irf3 @GGP@ {util.END_OF_REL_SYMBOL}"
    )
    actual = util.format_relation(
        ent_clusters=ent_clusters, ent_labels=ent_labels, rel_label=rel_label
    )
    assert actual == expected


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
    # A truncated example taken from the BC5CDR dataset
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

    title_clusters = {
        "D001569": schemas.PubtatorCluster(
            ents=["benzodiazepines"],
            offsets=[(28, 43)],
            label="Chemical",
        ),
    }
    abstract_clusters = {
        "D001569": schemas.PubtatorCluster(
            ents=["benzodiazepines", "BZDs"],
            offsets=[(219, 234), (253, 257)],
            label="Chemical",
        ),
        "D005221": schemas.PubtatorCluster(
            ents=["tiredness"], offsets=[(1817, 1826)], label="Disease"
        ),
    }
    both_clusters = {
        "D001569": schemas.PubtatorCluster(
            ents=["benzodiazepines", "BZDs"],
            offsets=[(28, 43), (253, 257)],
            label="Chemical",
        ),
        "D005221": schemas.PubtatorCluster(
            ents=["tiredness"], offsets=[(1817, 1826)], label="Disease"
        ),
    }

    # Title only
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=title_text,
            clusters=title_clusters,
            relations=[],
        ),
    }
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.title)
    # Breaking up the asserts leads to much clearer outputs when the test fails
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations

    # Abstract only
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=abstract_text,
            clusters=abstract_clusters,
            relations=[("D001569", "D005221", "CID")],
        ),
    }
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.abstract)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations

    # Both
    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=f"{title_text} {abstract_text}",
            clusters=both_clusters,
            relations=[("D001569", "D005221", "CID")],
        ),
    }
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations


def test_parse_pubtator_skip_malformed_raises_value_error() -> None:
    # A truncated example taken from the BC5CDR dataset
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
    # A truncated example taken from the BC5CDR dataset
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
    # A truncated example taken from the BC5CDR dataset
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

    expected = {
        pmid: schemas.PubtatorAnnotation(
            text=f"{title_text} {abstract_text}",
            clusters={
                "D019259": schemas.PubtatorCluster(
                    ents=["lamivudine"],
                    offsets=[(26, 36)],
                    label="Chemical",
                ),
                "D012964": schemas.PubtatorCluster(
                    ents=["na"], offsets=[(59, 61)], label="Chemical"
                ),
                "D006509": schemas.PubtatorCluster(
                    ents=["hepatitis B virus (HBV) infected", "HBV infected"],
                    offsets=[(66, 98), (186, 209)],
                    label="Disease",
                ),
                "D015658": schemas.PubtatorCluster(
                    ents=["HIV co-infection", "HIV infected"],
                    offsets=[(125, 141), (194, 209)],
                    label="Disease",
                ),
            },
        )
    }
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations
