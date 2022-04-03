import copy
import json

import pytest

from seq2rel_ds.common import schemas
from seq2rel_ds.common.special_tokens import COREF_SEP_SYMBOL, HINT_SEP_SYMBOL


def test_pubtator_entity_to_string() -> None:
    ent = schemas.PubtatorEntity(
        # Contains:
        # - multi-word mentions
        # - overlapping mentions
        # - multiple duplicate mentions
        # - at least two unique mentions (case-insensitive)
        # - mentions that are not already ordered by first appearance
        mentions=[
            "factor vii deficiency",
            "factor vii deficiency",
            "Factor VII Deficiency",
            "factor vii deficient",
        ],
        offsets=[(200, 221), (100, 121), (20, 41), (0, 21)],
        label="Disease",
    )
    # Test with sorting (which is and should be the default)
    actual = ent.to_string()
    expected = f"factor vii deficient {COREF_SEP_SYMBOL} factor vii deficiency @DISEASE@"
    assert actual == expected

    # Test without sorting
    # Note: because the mentions are randomly sorted when sort=False, we check a couple other
    # attributes, like length of the string.
    actual = ent.to_string(sort=False)
    assert len(actual) == len(expected)
    assert "factor vii deficient" in actual
    assert "factor vii deficiency" in actual
    assert "@DISEASE@" in actual
    assert COREF_SEP_SYMBOL in actual


def test_pubtator_entity_get_offsets() -> None:
    ent = schemas.PubtatorEntity(
        # We don't need actual mentions or a label to test this method.
        mentions=[
            "",
            "",
            "",
            "",
        ],
        offsets=[(200, 221), (100, 121), (20, 41), (0, 21)],
        label="",
    )
    expected = (0, 21)
    actual = ent.get_offsets()
    assert actual == expected


def test_insert_hints() -> None:
    """Asserts that insert_hints works as expected for a list of edge cases."""
    # A truncated example taken from the GDA dataset. It contains a few edge cases:
    # - coreferent mention
    # - entites that differ in case
    # - paranthesized entity
    # - multiple identical mentions of an entity
    text = (
        "Apolipoprotein E epsilon4 allele, elevated midlife total cholesterol level, and high"
        " midlife systolic blood pressure are independent risk factors for late-life Alzheimer disease."
        " BACKGROUND: Presence of the apolipoprotein E (apoE) epsilon4 allele, which is involved in"
        " cholesterol metabolism, is the most important genetic risk factor for Alzheimer disease."
        " Elevated midlife values for total cholesterol level and blood pressure have been"
        " implicated recently as risk factors for Alzheimer disease."
    )

    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="12160362",
        text=text,
        entities={
            "348": schemas.PubtatorEntity(
                mentions=["Apolipoprotein E", "apolipoprotein E", "apoE"],
                offsets=[(0, 17), (207, 223), (225, 229)],
                label="Gene",
            ),
            "D000544": schemas.PubtatorEntity(
                mentions=["Alzheimer disease", "Alzheimer disease", "Alzheimer disease"],
                offsets=[(160, 177), (339, 356), (479, 496)],
                label="Disease",
            ),
        },
    )
    expected = f"apolipoprotein e {COREF_SEP_SYMBOL} apoe @GENE@ alzheimer disease @DISEASE@ {HINT_SEP_SYMBOL} {text}"

    pubator_annotation.insert_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_hints_compound() -> None:
    """Asserts that insert_hints works as expected for compound entities."""
    text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " intrahepatic and extrahepatic cholestasis."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="9862868",
        text=text,
        entities={
            "D002780": schemas.PubtatorEntity(
                mentions=["intrahepatic cholestasis"], offsets=[(87, 128)], label="Disease"
            ),
            "D001651": schemas.PubtatorEntity(
                mentions=["extrahepatic cholestasis"], offsets=[(104, 128)], label="Disease"
            ),
        },
    )
    expected = f"intrahepatic cholestasis @DISEASE@ extrahepatic cholestasis @DISEASE@ {HINT_SEP_SYMBOL} {text}"

    pubator_annotation.insert_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_hints_overlapping() -> None:
    """Asserts that insert_hints works as expected for overlapping entities."""
    text = (
        "Mutation pattern in clinically asymptomatic coagulation factor VII deficiency. A total of"
        " 122 subjects, referred after presurgery screening or checkup for prolonged prothrombin"
        " time, were characterized for the presence of coagulation factor VII deficiency."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="8844208",
        text=text,
        entities={
            "2155": schemas.PubtatorEntity(
                mentions=["coagulation factor VII", "coagulation factor VII"],
                offsets=[(44, 67), (222, 244)],
                label="Gene",
            ),
            "D005168": schemas.PubtatorEntity(
                mentions=["factor VII deficiency", "factor VII deficiency"],
                offsets=[(56, 78), (234, 255)],
                label="Disease",
            ),
        },
    )
    expected = (
        f"coagulation factor vii @GENE@ factor vii deficiency @DISEASE@ {HINT_SEP_SYMBOL} {text}"
    )

    pubator_annotation.insert_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_hints_no_mutation() -> None:

    """Asserts that insert_hints does not mutate any attribute beside `text`."""
    text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " intrahepatic and extrahepatic cholestasis."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="9862868",
        text=text,
        entities={
            "D002780": schemas.PubtatorEntity(
                mentions=["intrahepatic cholestasis"], offsets=[(87, 128)], label="Disease"
            ),
            "D001651": schemas.PubtatorEntity(
                mentions=["extrahepatic cholestasis"], offsets=[(104, 128)], label="Disease"
            ),
        },
    )
    expected = copy.deepcopy(pubator_annotation)

    pubator_annotation.insert_hints()

    assert pubator_annotation.text != expected.text
    assert pubator_annotation.pmid == expected.pmid
    assert pubator_annotation.entities == expected.entities
    assert pubator_annotation.relations == expected.relations


def test_pubtator_annotation_to_string() -> None:
    # Contains:
    # - at least one entity with multiple mentions, including a unique mention
    # - at least two relations with different head entities
    # - at least one n-ary relation
    # - relations that are not already ordered by first appearance
    ann = schemas.PubtatorAnnotation(
        # We don't need text or a PMID to test this method.
        pmid="",
        text="",
        entities={
            "D008094": schemas.PubtatorEntity(
                mentions=["lithium", "lithium", "Li", "Li"],
                offsets=[(54, 61), (111, 118), (941, 943), (1333, 1335)],
                label="Chemical",
            ),
            "D006973": schemas.PubtatorEntity(
                mentions=["hypertension", "hypertension"],
                offsets=[(1000, 1012), (1500, 1512)],
                label="Disease",
            ),
            "D011507": schemas.PubtatorEntity(
                mentions=["proteinuria", "proteinuria"],
                offsets=[(975, 986), (1466, 1477)],
                label="Disease",
            ),
            "D007676": schemas.PubtatorEntity(
                mentions=["chronic renal failure", "chronic renal failure"],
                offsets=[(70, 91), (1531, 1552)],
                label="Disease",
            ),
        },
        relations=[
            ("D008094", "D006973", "CID"),
            ("D008094", "D011507", "CID"),
            ("D008094", "D007676", "CID"),
            # This is an artificial n-ary relation.
            ("D008094", "D006973", "D011507", "CID"),
        ],
    )

    # Test with sorting (which is and should be the default)
    actual = ann.to_string()
    expected = (
        f"lithium {COREF_SEP_SYMBOL} li @CHEMICAL@ chronic renal failure @DISEASE@ @CID@"
        f" lithium {COREF_SEP_SYMBOL} li @CHEMICAL@ proteinuria @DISEASE@ @CID@"
        f" lithium {COREF_SEP_SYMBOL} li @CHEMICAL@ hypertension @DISEASE@ @CID@"
        f" lithium {COREF_SEP_SYMBOL} li @CHEMICAL@ hypertension @DISEASE@ proteinuria @DISEASE@ @CID@"
    )
    assert actual == expected

    # Test without sorting
    # Note: because the mentions are randomly sorted when sort=False, we check a couple other
    # attributes, like length of the string.
    actual = ann.to_string(sort=False)
    assert len(actual) == len(expected)
    assert "lithium" in actual
    assert "li" in actual
    assert "chronic renal failure" in actual
    assert "proteinuria" in actual
    assert "hypertension" in actual
    assert "@CHEMICAL@" in actual
    assert "@DISEASE@" in actual
    assert "@CID@" in actual
    assert COREF_SEP_SYMBOL in actual


def test_pydantic_encoder(dummy_annotation_pydantic) -> None:
    # This is a pretty shallow test, but it will throw an error if the object cannot be serialized.
    _ = json.dumps(dummy_annotation_pydantic, indent=2, cls=schemas.PydanticEncoder)


def test_pydantic_encoder_type_error() -> None:
    # Exactly what we pass here doesn't matter, so long as it is not JSON seriablizable.
    non_serializable = {"arbitrary": 3 + 6j}
    with pytest.raises(TypeError):
        _ = json.dumps(non_serializable, indent=2, cls=schemas.PydanticEncoder)


def test_as_pubtator_annotation(dummy_annotation_json) -> None:
    actual = json.loads(dummy_annotation_json, object_hook=schemas.as_pubtator_annotation)
    assert isinstance(actual.pmid, str)
    assert isinstance(actual, schemas.PubtatorAnnotation)
    for uid, ent in actual.entities.items():
        assert isinstance(uid, str)
        assert isinstance(ent, schemas.PubtatorEntity)
