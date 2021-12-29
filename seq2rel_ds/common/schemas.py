import copy
import json
import random
from operator import itemgetter
from typing import Dict, List, Tuple

from pydantic import BaseModel
from seq2rel_ds.common import sorting_utils, special_tokens
from seq2rel_ds.common.text_utils import sanitize_text


class AlignedExample(BaseModel):
    doc_id: str
    text: str
    relations: str
    score: float


class PubtatorCluster(BaseModel):
    """A Pydantic model for storing entity annotations."""

    mentions: List[str]
    offsets: List[Tuple[int, int]]
    label: str

    def to_string(self, sort: bool = True) -> str:
        """Linearizes this entity's mentions, returning a string. Removes duplicate mentions
        (case-insensitive). If `sort=True`, mentions are sorted by their order of first appearance.
        """
        mentions = copy.deepcopy(self.mentions)
        # Optionally, sort by order of first appearance using the end character offsets.
        # This exists mainly for ablation, so we randomly shuffle mentions if sort=False.
        offsets = [end for _, end in self.offsets]
        if sort:
            mentions, _ = sorting_utils.sort_by_offset(mentions, offsets)
        else:
            random.shuffle(mentions)
        # Remove duplicates (case-insensitive) but maintain order.
        mentions = [sanitize_text(m, lowercase=True) for m in mentions]
        mentions = list(dict.fromkeys(mentions))
        # Create the linearized entity string.
        mention_string = f" {special_tokens.COREF_SEP_SYMBOL} ".join(mentions)
        entity_string = f"{mention_string.strip().lower()} @{self.label.strip().upper()}@"
        return entity_string

    def get_offset(self) -> int:
        """Returns the sum of the start and end character offsets of the first occurring mention of
        this entity."""
        return sum(min(self.offsets, key=itemgetter(1)))


class PubtatorAnnotation(BaseModel):
    """A Pydantic model for storing relation annotations. The last item in `relations` is the
    class label.
    """

    text: str
    pmid: str
    clusters: Dict[str, PubtatorCluster] = {}
    relations: List[Tuple[str, ...]] = []

    def insert_hints(self, sort: bool = True) -> None:
        """Inserts entity hints into the beginning of `self.text`. This effectively turns the
        task into relation extraction (as opposed to joint entity and relation extraction).
        """
        entity_strings = [ent.to_string() for ent in self.clusters.values()]
        # Sort by the character end offset of the first mention.
        entity_offsets = [ent.get_offset() for ent in self.clusters.values()]
        # Optionally, sort by order of first appearance using the end character offsets.
        # This exists mainly for ablation, so we randomly shuffle entities if sort=False.
        if sort:
            entity_strings, _ = sorting_utils.sort_by_offset(entity_strings, entity_offsets)
        else:
            random.shuffle(entity_strings)
        # Remove duplicates but maintain order.
        entity_strings = list(dict.fromkeys(entity_strings))
        # Create the linearized entity hint and insert it at the beggining of the source text.
        entity_hint = f'{" ".join(entity_strings).strip()} {special_tokens.HINT_SEP_SYMBOL}'
        self.text = f"{entity_hint.strip()} {self.text.strip()}"

    def to_string(self, sort: bool = True) -> str:
        """Linearizes the relations, returning a string. If `sort=True`, relations are sorted by
        their order of first appearance.
        """
        relation_strings = []
        relation_offsets = []
        for rel in self.relations:
            entity_strings, relation_offset = [], 0
            for ent_id in rel[:-1]:
                ent = self.clusters[ent_id]
                entity_strings.append(ent.to_string())
                relation_offset += ent.get_offset()
            relation_string = sanitize_text(f'{" ".join(entity_strings)} @{rel[-1].upper()}@')
            relation_strings.append(relation_string)
            relation_offsets.append(relation_offset)
        # Optionally, sort by order of first appearance.
        # This exists mainly for ablation, so we randomly shuffle relations if sort=False.
        if sort:
            # We may encounter relations with identical entities but in a different order or with a
            # different relation type. To handle this case, first sort relation strings
            # lexographically and then by offset.
            relation_strings, relation_offsets = sorting_utils.sort_by_offset(
                relation_strings, relation_offsets, key=None
            )
            relation_strings, _ = sorting_utils.sort_by_offset(relation_strings, relation_offsets)
        else:
            random.shuffle(relation_strings)
        # Remove duplicates but maintain order.
        relation_strings = list(dict.fromkeys(relation_strings))
        # Create the linearized relation string
        relation_string = " ".join(relation_strings).strip()
        return relation_string


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
