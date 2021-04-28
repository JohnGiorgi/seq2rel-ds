import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from seq2rel_ds.common import util

app = typer.Typer()

TRAIN_ANNOTATED_FILENAME = "train_annotated.json"
TRAIN_DISTANT_FILENAME = "train_distant.json"
VALID_FILENAME = "dev.json"
TEST_FILENAME = "test.json"
REL_INFO_FILENAME = "rel_info.json"

# Some custom types for working with the DocRED vertexSet JSON object
VertexSet = List[Dict[str, Any]]
ParsedVetex = Tuple[List[str], List[List[int]], str]


def _parse_vertex_set(vertex_set: VertexSet) -> ParsedVetex:
    names: List[str] = []
    offsets: List[List[int]] = []
    label: str = ""
    for vertex in vertex_set:
        name = vertex["name"].lower()
        if name not in names:
            names.append(name)
        offsets.append(vertex["pos"])
        if not label:
            label = vertex["type"]
    return names, offsets, label


def _preprocess(filepath: Path, rel_labels: Optional[Dict[str, str]] = None) -> List[str]:
    dataset = json.loads(filepath.read_text().strip())
    processed_dataset = []
    for example in dataset:
        text = " ".join([" ".join(sent) for sent in example["sents"]])
        relations = []
        offsets = []
        # The use of get here is because the key "labels" may not exist for some examples
        for label in example.get("labels", []):
            # Rel labels are just arbitrary identifiers. Use the provided dict to get actual names
            rel_label = (
                "_".join(rel_labels[label["r"]].split()).upper() if rel_labels else label["r"]
            )
            # For some godforsaken reason, the indices are 1-indexed
            head_vertex = example["vertexSet"][label["h"]]
            tail_vertex = example["vertexSet"][label["t"]]

            head_names, head_offsets, head_label = _parse_vertex_set(head_vertex)
            tail_names, tail_offsets, tail_label = _parse_vertex_set(tail_vertex)

            # Keep track of the end offsets of each entity. We will use these to sort
            # relations according to their order of first appearence in the text.
            head_offset = min((end for _, end in head_offsets))
            tail_offset = min((end for _, end in head_offsets))
            offset = head_offset + tail_offset

            ent_clusters = [head_names, tail_names]
            ent_labels = [head_label, tail_label]

            relation = util.format_relation(
                ent_clusters=ent_clusters,
                ent_labels=ent_labels,
                rel_label=rel_label,
            )
            relations.append(relation)
            offsets.append(offset)

        if relations:
            relations = util.sort_by_offset(relations, offsets)
        processed_dataset.append(f"{text}\t{' '.join(relations)}")
    return processed_dataset


@app.callback(invoke_without_command=True)
def main(
    input_dir: Path = typer.Argument(..., help="Path to a local copy of the DocRED corpus."),
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
    use_distant: bool = typer.Option(
        False,
        help=(
            "Pass this to use the distantly supervised train data."
            " Otherwise, the annotated train data is used."
        ),
    ),
) -> None:
    """Preprocess a local copy of the DocRED corpus for use with seq2rel."""
    train_filename = TRAIN_DISTANT_FILENAME if use_distant else TRAIN_ANNOTATED_FILENAME
    train_filepath = Path(input_dir) / train_filename
    dev_filepath = Path(input_dir) / VALID_FILENAME
    test_filepath = Path(input_dir) / TEST_FILENAME

    rel_info = json.loads((Path(input_dir) / REL_INFO_FILENAME).read_text())

    train = _preprocess(train_filepath, rel_labels=rel_info)
    valid = _preprocess(dev_filepath, rel_labels=rel_info)
    test = _preprocess(test_filepath, rel_labels=rel_info)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))


if __name__ == "__main__":
    app()
