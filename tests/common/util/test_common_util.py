import random
import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import booleans, text
from seq2rel_ds.common import schemas, util


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


def test_train_valid_test_split():
    data = random.sample(range(0, 100), 10)
    train_size, valid_size, test_size = 0.7, 0.1, 0.2
    train, valid, test = util.train_valid_test_split(
        data, train_size=train_size, valid_size=valid_size, test_size=test_size
    )
    assert len(train) == int(train_size * len(data))
    assert len(valid) == int(valid_size * len(data))
    assert len(test) == int(test_size * len(data))


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


def test_parse_pubtator_raises_value_error() -> None:
    pmid = "2339463"
    title_text = "Cerebral sinus thrombosis as a potential hazard of antifibrinolytic treatment in menorrhagia."
    abstract_text = (
        "We describe a 42-year-old woman who developed superior sagittal and left transverse sinus"
        " thrombosis associated with prolonged epsilon-aminocaproic acid therapy for menorrhagia."
    )
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}	0	25	Cerebral sinus thrombosis	Disease
    """

    # There should be no error when skip_malformed is True...
    _ = util.parse_pubtator(pubtator_content, skip_malformed=True)
    # ...and and error when it is False
    with pytest.raises(ValueError):
        _ = util.parse_pubtator(pubtator_content, skip_malformed=False)


def test_parse_pubtator() -> None:
    # An example taken from the BC5CDR dataset
    pmid = "2339463"
    title_text = "Cerebral sinus thrombosis as a potential hazard of antifibrinolytic treatment in menorrhagia."
    abstract_text = (
        "We describe a 42-year-old woman who developed superior sagittal and left transverse sinus"
        " thrombosis associated with prolonged epsilon-aminocaproic acid therapy for menorrhagia."
    )
    pubtator_content = f"""
    {pmid}|t|{title_text}
    {pmid}|a|{abstract_text}
    {pmid}	0	25	Cerebral sinus thrombosis	Disease	D012851
    {pmid}	81	92	menorrhagia	Disease	D008595
    {pmid}	149	194	sagittal and left transverse sinus thrombosis	Disease	D020225|D020227	sagittal sinus thrombosis|left transverse sinus thrombosis
    {pmid}	221	246	epsilon-aminocaproic acid	Chemical	D015119
    {pmid}	CID	D015119	D020225
    """

    title_clusters = {
        "D012851": schemas.PubtatorCluster(
            ents=["cerebral sinus thrombosis"], offsets=[(0, 25)], label="Disease"
        ),
        "D008595": schemas.PubtatorCluster(
            ents=["menorrhagia"], offsets=[(81, 92)], label="Disease"
        ),
    }
    abstract_clusters = {
        "D020225": schemas.PubtatorCluster(
            ents=["sagittal sinus thrombosis"], offsets=[(149, 194)], label="Disease"
        ),
        "D020227": schemas.PubtatorCluster(
            ents=["left transverse sinus thrombosis"], offsets=[(149, 194)], label="Disease"
        ),
        "D015119": schemas.PubtatorCluster(
            ents=["epsilon-aminocaproic acid"], offsets=[(221, 246)], label="Chemical"
        ),
    }
    both_clusters = {
        "D012851": schemas.PubtatorCluster(
            ents=["cerebral sinus thrombosis"], offsets=[(0, 25)], label="Disease"
        ),
        "D008595": schemas.PubtatorCluster(
            ents=["menorrhagia"], offsets=[(81, 92)], label="Disease"
        ),
        "D020225": schemas.PubtatorCluster(
            ents=["sagittal sinus thrombosis"], offsets=[(149, 194)], label="Disease"
        ),
        "D020227": schemas.PubtatorCluster(
            ents=["left transverse sinus thrombosis"], offsets=[(149, 194)], label="Disease"
        ),
        "D015119": schemas.PubtatorCluster(
            ents=["epsilon-aminocaproic acid"], offsets=[(221, 246)], label="Chemical"
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
            relations=[("D015119", "D020225", "CID")],
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
            relations=[("D015119", "D020225", "CID")],
        ),
    }
    actual = util.parse_pubtator(pubtator_content, text_segment=util.TextSegment.both)
    assert actual[pmid].text == expected[pmid].text
    assert actual[pmid].clusters == expected[pmid].clusters
    assert actual[pmid].relations == expected[pmid].relations
