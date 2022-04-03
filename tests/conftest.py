import copy
import json
from typing import Any, Dict, List

import pytest

from seq2rel_ds.common.schemas import PubtatorAnnotation


@pytest.fixture
def dummy_annotation_dict() -> Dict[str, Any]:
    annotation = {
        "pmid": "2038333",
        "text": (
            "Mutations of yeast CYC8 or TUP1 genes greatly reduce the degree of glucose repression"
            " of many genes and affect other regulatory pathways, including mating type."
        ),
        "entities": {
            "852410": {"mentions": ["cyc8"], "offsets": [[142, 146]], "label": "Gene"},
            "850445": {"mentions": ["tup1"], "offsets": [[150, 154]], "label": "Gene"},
        },
        "relations": [],
    }

    return annotation


@pytest.fixture()
def dummy_annotation_json(dummy_annotation_dict) -> str:
    return json.dumps(dummy_annotation_dict)


@pytest.fixture()
def dummy_annotation_pydantic(dummy_annotation_dict) -> List[PubtatorAnnotation]:
    annotation = copy.deepcopy(dummy_annotation_dict)
    return PubtatorAnnotation(**annotation)
