from typing import List, Tuple, Dict
import json

from pydantic import BaseModel


class AlignedExample(BaseModel):
    doc_id: str
    text: str
    relations: str
    score: float


class PubtatorCluster(BaseModel):
    ents: List[str]
    offsets: List[Tuple[int, int]]
    label: str


class PubtatorAnnotation(BaseModel):
    text: str
    clusters: Dict[str, PubtatorCluster] = {}
    relations: List[Tuple[str, ...]] = []


class PydanticEncoder(json.JSONEncoder):
    """A custom encoder so we can call `json` methods on objects with Pydantic models in them

    See: https://docs.python.org/3/library/json.html for more details.
    """

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def as_pubtator_annotation(dct):
    """To be used as `object_hook` with `json` methods to load serialized data as
    `PubtatorAnnotation` objects.
    """
    if "text" in dct and "clusters" in dct and "relations" in dct:
        return PubtatorAnnotation(**dct)
    return dct
