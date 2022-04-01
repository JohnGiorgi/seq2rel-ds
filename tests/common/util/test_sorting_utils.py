from typing import List, Tuple

import pytest

from seq2rel_ds.common import sorting_utils


def test_pubtator_ann_is_mention() -> None:
    assert sorting_utils.pubtator_ann_is_mention(
        "12119460\t10\t24\t5-fluorouracil\tChemical\tD005472"
    )
    assert not sorting_utils.pubtator_ann_is_mention("12119460\tCID\tD002955\tD014839")
    assert not sorting_utils.pubtator_ann_is_mention(
        "12119460\tCID\tD002955\tD014839\n".split("\t")
    )


def test_sort_entity_annotations() -> None:
    # Test an empty (but iterable) input
    assert sorting_utils.sort_entity_annotations([]) == []
    assert sorting_utils.sort_entity_annotations(
        [
            # Purposefully but the relation first to confirm the
            # function places it at the end of the returned list.
            "12119460\tCID\tD002955\tD014839",
            "12119460\t27\t39\tfolinic acid\tChemical\tD002955",
            "12119460\t10\t24\t5-fluorouracil\tChemical\tD005472",
        ]
    ) == [
        "12119460\t10\t24\t5-fluorouracil\tChemical\tD005472",
        "12119460\t27\t39\tfolinic acid\tChemical\tD002955",
        "12119460\tCID\tD002955\tD014839",
    ]


def test_sort_by_offset() -> None:
    items = []
    offsets = []

    # Check that this is an no-op if items is empty.
    expected: Tuple[List[str], List[int]] = ([], [])
    actual = sorting_utils.sort_by_offset([], [])

    assert actual == expected

    items = ["b", "c", "a"]
    offsets = [1, 0, 2]

    expected = (["c", "b", "a"], [0, 1, 2])
    actual = sorting_utils.sort_by_offset(items, offsets)

    assert actual == expected
    # Check that we did not mutate the input
    assert items == ["b", "c", "a"]


def test_sort_by_offset_with_key() -> None:
    items = []
    offsets = []

    # Check that this is an no-op if items is empty.
    expected: Tuple[List[str], List[int]] = ([], [])
    actual = sorting_utils.sort_by_offset([], [], key=None)

    assert actual == expected

    items = ["b", "c", "a"]
    offsets = [1, 0, 2]

    expected = (["a", "b", "c"], [2, 1, 0])
    actual = sorting_utils.sort_by_offset(items, offsets, key=None)

    assert actual == expected
    # Check that we did not mutate the input
    assert items == ["b", "c", "a"]


def test_sort_by_offset_raise_value_error() -> None:
    items = ["b", "c", "a"]
    offsets = [1, 2]

    with pytest.raises(ValueError):
        _ = sorting_utils.sort_by_offset(items, offsets)
