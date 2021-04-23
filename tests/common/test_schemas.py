from seq2rel_ds.common import schemas
import json
import pytest


def test_pydantic_encoder(dummy_annotation_pydantic) -> None:
    # This is a pretty shallow test, but it will throw an error if the object cannot be serialized.
    _ = json.dumps(dummy_annotation_pydantic, indent=2, cls=schemas.PydanticEncoder)


def test_pydantic_encoder_type_error(dummy_annotation_pydantic) -> None:
    # Exactly what we pass here doesn't matter, so long as it is not JSON seriablizable.
    non_serializable = {"arbitrary": 3 + 6j}
    with pytest.raises(TypeError):
        _ = json.dumps(non_serializable, indent=2, cls=schemas.PydanticEncoder)


def test_as_pubtator_annotation(dummy_annotation_json) -> None:
    actual = json.loads(dummy_annotation_json, object_hook=schemas.as_pubtator_annotation)
    for pmid, annotation in actual.items():
        assert isinstance(pmid, str)
        assert isinstance(annotation, schemas.PubtatorAnnotation)
        for uid, cluster in annotation.clusters.items():
            assert isinstance(uid, str)
            assert isinstance(cluster, schemas.PubtatorCluster)
