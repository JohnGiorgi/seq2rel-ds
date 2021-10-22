from typing import List, Tuple, Dict
import json

from pydantic import BaseModel
from seq2rel_ds.common import special_tokens, sorting_utils


class AlignedExample(BaseModel):
    doc_id: str
    text: str
    relations: str
    score: float


class PubtatorCluster(BaseModel):
    """A Pydantic model for storing coreferent entity annotations."""

    mentions: List[str]
    offsets: List[Tuple[int, int]]
    label: str

    def unique_mentions(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Returns a list of tuples containing the unique entity mentions (case-insensitive) and
        their character offsets.
        """
        unique_mentions: List[str] = []
        unique_offsets: List[Tuple[int, int]] = []
        for mention, offsets in zip(self.mentions, self.offsets):
            if mention.lower() not in [m.lower() for m in unique_mentions]:
                unique_mentions.append(mention)
                unique_offsets.append(offsets)
        return unique_mentions, unique_offsets


class PubtatorAnnotation(BaseModel):
    """A Pydantic model for storing relation annotations. The last item in `relations` is the
    class label.
    """

    text: str
    pmid: str
    clusters: Dict[str, PubtatorCluster] = {}
    relations: List[Tuple[str, ...]] = []

    def insert_entity_hints(self) -> None:
        """Adds entity hints to the beginning of `self.text`. This effectively turns the task into
        relation extraction (as opposed to joint entity and relation extraction).
        """
        formatted_ents, ent_offsets = [], []
        for ent in self.clusters.values():
            mentions, mention_offsets = ent.unique_mentions()
            end_offsets = [end for _, end in mention_offsets]
            mentions = sorting_utils.sort_by_offset(mentions, end_offsets)
            formatted_mentions = f" {special_tokens.COREF_SEP_SYMBOL} ".join(mentions).strip()
            formatted_ents.append(f"{formatted_mentions} @{ent.label.upper()}@")
            ent_offsets.append(min(end_offsets))

        formatted_ents = sorting_utils.sort_by_offset(formatted_ents, ent_offsets)
        ent_hint = f"{' '.join(formatted_ents).strip()} {special_tokens.HINT_SEP_SYMBOL} "
        self.text = ent_hint + self.text.strip()


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
    if all(key in dct for key in ["text", "pmid", "clusters", "relations"]):
        return PubtatorAnnotation(**dct)
    return dct
