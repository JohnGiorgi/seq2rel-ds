from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import typer

from seq2rel_ds import msg
from seq2rel_ds.common import text_utils, util

app = typer.Typer()

DOCRED_URL = "http://lavis.cs.hs-rm.de/storage/jerex/public/datasets/docred_joint/"
TRAIN_FILENAME = "train_joint.json"
VALID_FILENAME = "dev_joint.json"
TEST_FILENAME = "test_joint.json"
TYPES_FILENAME = "types.json"

# Some custom types for working with the DocRED vertexSet JSON object
VertexSet = List[Dict[str, Any]]
ParsedVetex = Tuple[List[str], List[List[int]], str]


def _download_corpus() -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    train = requests.get(f"{DOCRED_URL}/{TRAIN_FILENAME}").json()
    valid = requests.get(f"{DOCRED_URL}/{VALID_FILENAME}").json()
    test = requests.get(f"{DOCRED_URL}/{TEST_FILENAME}").json()
    types = requests.get(f"{DOCRED_URL}/{TYPES_FILENAME}").json()

    return train, valid, test, types


def _convert_to_pubtator(
    examples: List[Dict[str, Any]], rel_labels: Optional[Dict[str, str]] = None
) -> str:
    pubtator_formatted_anns = []
    for doc_id, example in enumerate(examples):
        sents = example["sents"]
        text = text_utils.sanitize_text(" ".join(" ".join(sent) for sent in sents))
        pubtator_formatted_ann = f"{doc_id}|t|\n{doc_id}|a|{text}\n"
        for ent_id, ent in enumerate(example["vertexSet"]):
            for mention in ent:
                start, end = mention["pos"]
                name = text_utils.sanitize_text(mention["name"])
                type_ = mention["type"]
                # The start and end indexes are relative to their own sentence.
                # Account for this to produce document-level offsets.
                sent_offset = sum(len(sent) for sent in sents[: mention["sent_id"]])
                start += sent_offset
                end += sent_offset

                pubtator_formatted_ann += f"{doc_id}\t{start}\t{end}\t{name}\t{type_}\t{ent_id}\n"

        for rel in example.get("labels", []):
            label = rel["r"]
            head = rel["h"]
            tail = rel["t"]

            # If the `rel_labels` dict was provided, get the verbose or human readable label.
            if rel_labels:
                label = "_".join(rel_labels[label].strip().replace(",", "").upper().split())

            pubtator_formatted_ann += f"{doc_id}\t{label}\t{head}\t{tail}\n"

        pubtator_formatted_anns.append(pubtator_formatted_ann.strip())

    return "\n\n".join(pubtator_formatted_anns)


def _preprocess(
    examples: List[Dict[str, Any]],
    rel_labels: Optional[Dict[str, str]] = None,
    sort_rels: bool = True,
) -> List[str]:
    pubtator_content = _convert_to_pubtator(examples, rel_labels=rel_labels)
    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content,
        text_segment=util.TextSegment.abstract,
    )
    seq2rel_annotations = util.pubtator_to_seq2rel(pubtator_annotations, sort_rels=sort_rels)

    return seq2rel_annotations


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
    sort_rels: bool = typer.Option(
        True, help="Sort relations according to order of first appearance."
    ),
) -> None:
    """Download and preprocess the DocRED corpus for use with seq2rel."""
    msg.divider("Preprocessing DocRED")

    with msg.loading("Downloading corpus..."):
        train_raw, valid_raw, test_raw, types = _download_corpus()
    msg.good("Downloaded the corpus.")

    # Create a dictionary mapping relation labels to their actual names.
    rel_labels = {key: value["verbose"] for key, value in types["relations"].items()}

    with msg.loading("Preprocessing the data..."):
        train = _preprocess(train_raw, rel_labels=rel_labels, sort_rels=sort_rels)
        valid = _preprocess(valid_raw, rel_labels=rel_labels, sort_rels=sort_rels)
        test = _preprocess(test_raw, rel_labels=rel_labels, sort_rels=sort_rels)
    msg.good("Preprocessed the data.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}.")


if __name__ == "__main__":
    app()
