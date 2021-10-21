import copy
import json

import pytest
from seq2rel_ds.common import schemas
from seq2rel_ds.common.special_tokens import COREF_SEP_SYMBOL, HINT_SEP_SYMBOL


def test_insert_entity_hints() -> None:
    """Asserts that insert_entity_hints works as expected for a list of edge cases."""
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
        clusters={
            "348": schemas.PubtatorCluster(
                ents=["Apolipoprotein E", "apolipoprotein E", "apoE"],
                offsets=[(0, 17), (207, 223), (225, 229)],
                label="Gene",
            ),
            "D000544": schemas.PubtatorCluster(
                ents=["Alzheimer disease", "Alzheimer disease", "Alzheimer disease"],
                offsets=[(160, 177), (339, 356), (479, 496)],
                label="Disease",
            ),
        },
    )
    expected = f"Apolipoprotein E {COREF_SEP_SYMBOL} apoE @GENE@ Alzheimer disease @DISEASE@ {HINT_SEP_SYMBOL} {text}"

    pubator_annotation.insert_entity_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_entity_hints_compound() -> None:
    """Asserts that insert_entity_hints works as expected for compound entities."""
    text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " intrahepatic and extrahepatic cholestasis."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="9862868",
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
    expected = f"intrahepatic cholestasis @DISEASE@ extrahepatic cholestasis @DISEASE@ {HINT_SEP_SYMBOL} {text}"

    pubator_annotation.insert_entity_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_entity_hints_overlapping() -> None:
    """Asserts that insert_entity_hints works as expected for overlapping entities."""
    text = (
        "Mutation pattern in clinically asymptomatic coagulation factor VII deficiency. A total of"
        " 122 subjects, referred after presurgery screening or checkup for prolonged prothrombin"
        " time, were characterized for the presence of coagulation factor VII deficiency."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="8844208",
        text=text,
        clusters={
            "2155": schemas.PubtatorCluster(
                ents=["coagulation factor VII", "coagulation factor VII"],
                offsets=[(44, 67), (222, 244)],
                label="Gene",
            ),
            "D005168": schemas.PubtatorCluster(
                ents=["factor VII deficiency", "factor VII deficiency"],
                offsets=[(56, 78), (234, 255)],
                label="Disease",
            ),
        },
    )
    expected = (
        f"coagulation factor VII @GENE@ factor VII deficiency @DISEASE@ {HINT_SEP_SYMBOL} {text}"
    )

    pubator_annotation.insert_entity_hints()
    actual = pubator_annotation.text

    assert actual == expected


def test_insert_entity_hints_no_mutation() -> None:

    """Asserts that insert_entity_hints does not mutate any attribute beside `text`."""
    text = (
        "Different lobular distributions of altered hepatocyte tight junctions in rat models of"
        " intrahepatic and extrahepatic cholestasis."
    )
    pubator_annotation = schemas.PubtatorAnnotation(
        pmid="9862868",
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

    pubator_annotation.insert_entity_hints()

    assert pubator_annotation.text != expected.text
    assert pubator_annotation.pmid == expected.pmid
    assert pubator_annotation.clusters == expected.clusters
    assert pubator_annotation.relations == expected.relations


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
    for uid, cluster in actual.clusters.items():
        assert isinstance(uid, str)
        assert isinstance(cluster, schemas.PubtatorCluster)
