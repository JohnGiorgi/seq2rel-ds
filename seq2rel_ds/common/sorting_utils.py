from operator import itemgetter
from typing import Any, List, Tuple, Union


def pubtator_ann_is_mention(pubtator_ann: Union[str, List[str]]) -> bool:
    """Return True if a `pubtator_ann` is an entity mention annotation and False if it is a relation
    annotation. `pubtator_ann` can be provided as a tab-delimited string or a list of strings.

    Preconditions:
        - `pubtator_ann` is a valid PubTator formatted annotation and is not a title or abstract line.
    """
    if isinstance(pubtator_ann, str):
        pubtator_ann = pubtator_ann.strip().split("\t")
    # If the second and third items are integers, this is an entity mention annotation.
    try:
        _ = int(pubtator_ann[1])
        _ = int(pubtator_ann[2])
        return True
    # Otherwise, it is a relation annotation.
    except ValueError:
        return False


def sort_entity_annotations(annotations: List[str], **kwargs: Any) -> List[str]:
    """Sort PubTator entity annotations by order of first appearence. Optional `**kwargs` are
    passed to `sorted`.
    """
    # We only sort the entities, so we have to seperate them from the relations,
    # perform the sort and then join everything together.
    ents = [ann for ann in annotations if pubtator_ann_is_mention(ann)]
    rels = [ann for ann in annotations if not pubtator_ann_is_mention(ann)]
    sorted_ents = sorted(ents, key=lambda x: int(x.split("\t")[2]), **kwargs)
    return sorted_ents + rels


def sort_by_offset(
    items: List[Any], offsets: List[int], **kwargs: Any
) -> Tuple[List[Any], List[int]]:
    """Returns `items`, sorted in ascending order according to `offsets`. Raises a `ValueError`
    if `len(items) != len(offsets)`. If `items` is empty, this is a no-op. Optional `**kwargs` are
    passed to `sorted`.
    """
    if len(items) != len(offsets):
        raise ValueError(f"len(items) ({len(items)}) != len(offsets) ({len(offsets)})")
    if not items:
        return items, offsets
    packed = list(zip(items, offsets))
    key = kwargs.pop("key", itemgetter(1))
    packed = sorted(packed, key=key, **kwargs)
    sorted_items, sorted_offsets = list(zip(*packed))
    sorted_items, sorted_offsets = list(sorted_items), list(sorted_offsets)
    return sorted_items, sorted_offsets
