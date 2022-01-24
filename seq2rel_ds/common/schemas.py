import copy
import json
import random
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
        offsets = copy.deepcopy(self.offsets)
        # Optionally, sort by order of first appearance using the end character offsets.
        # This exists mainly for ablation, so we randomly shuffle mentions if sort=False.
        if sort:
            mentions, _ = sorting_utils.sort_by_offset(mentions, offsets, key=lambda x: sum(x[1]))
        else:
            random.shuffle(mentions)
        # Remove duplicates (case-insensitive) but maintain order.
        mentions = [sanitize_text(m, lowercase=True) for m in mentions]
        mentions = list(dict.fromkeys(mentions))
        # Create the linearized entity string.
        mention_string = f" {special_tokens.COREF_SEP_SYMBOL} ".join(mentions)
        entity_string = f"{mention_string.strip().lower()} @{self.label.strip().upper()}@"
        return entity_string

    def get_offsets(self) -> Tuple[int, ...]:
        """Returns start and end character offsets of the first occurring mention of this entity."""
        return min(self.offsets, key=lambda x: sum(x))


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
        # Optionally, sort by order of first appearance using the end character offsets.
        # This exists mainly for ablation, so we randomly shuffle entities if sort=False.
        if sort:
            entity_offsets = [ent.get_offsets() for ent in self.clusters.values()]
            entity_strings, _ = sorting_utils.sort_by_offset(
                entity_strings, entity_offsets, key=lambda x: sum(x[1])
            )
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
            # Everything but the last item in a relation tuple is an entity, hence [:-1]
            entity_strings = [self.clusters[ent_id].to_string() for ent_id in rel[:-1]]
            relation_string = sanitize_text(f'{" ".join(entity_strings)} @{rel[-1].upper()}@')
            entity_offsets = [sum(self.clusters[ent_id].get_offsets()) for ent_id in rel[:-1]]
            relation_strings.append(relation_string)
            relation_offsets.append(entity_offsets)

        # Optionally, sort by order of first appearance.
        # This exists mainly for ablation, so we randomly shuffle relations if sort=False.
        if relation_strings:
            if sort:
                # Relations are sorted by their order of first appearance in the text. To determine
                # order, we use the character offsets of a relations entities. We first sort
                # according to the head entities, then the tail entities (and so-on for n-ary).
                # Functionally, this is equivalent to first sorting by the sum of entity offsets,
                # and then the n - 1 entity offsets.
                relation_strings, relation_offsets = sorting_utils.sort_by_offset(
                    relation_strings, relation_offsets, key=lambda x: sum(x[1])
                )
                for i in range(len(relation_offsets[0]) - 1):
                    relation_strings, relation_offsets = sorting_utils.sort_by_offset(
                        relation_strings, relation_offsets, key=lambda x: x[1][i]
                    )
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
