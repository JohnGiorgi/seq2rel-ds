from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from datasets import load_dataset, Dataset
from seq2rel_ds import msg
from seq2rel_ds.common import util

app = typer.Typer()


REL_LABEL = "ADE"
DRUG_LABEL = "DRUG"
EFFECT_LABEL = "EFFECT"


def get_offsets(text: str, drug: str, effect: str, indexes: Dict[str, Any]) -> Tuple[int, ...]:
    """Returns the start character index of `drug` and `effect` in `text`."""

    # Try to use the indices if they are provided.
    try:
        drug_start = indexes["drug"]["start_char"][0]
        drug_end = indexes["drug"]["end_char"][0]
    # Otherwise, take the first appearence in the text.
    except IndexError:
        drug_start = text.find(drug)
        drug_end = drug_start + len(drug)
    try:
        effect_start = indexes["effect"]["start_char"][0]
        effect_end = indexes["effect"]["end_char"][0]
    except IndexError:
        effect_start = text.find(effect)
        effect_end = effect_start + len(effect)

    return drug_start, drug_end, effect_start, effect_end


def _preprocess(dataset: Dataset) -> List[str]:
    dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")["train"]

    # Step 1: Process the dataset line by line, collecting formatted relations and their offsets.
    preprocessed_dataset: Dict[str, Any] = {}
    for line in dataset:
        text = line["text"]
        drug = line["drug"]
        effect = line["effect"]
        indexes = line["indexes"]

        relation = util.format_relation(
            ent_clusters=[[drug]] + [[effect]],
            ent_labels=[DRUG_LABEL, EFFECT_LABEL],
            rel_label=REL_LABEL,
        )
        # We take the relations position to be the sum of end indices of its entities.
        _, drug_end, _, effect_end = get_offsets(text, drug, effect, indexes)
        offset = drug_end + effect_end

        # Don't retain relations that are otherwise identical except for entity offsets or case.
        if text in preprocessed_dataset:
            if relation.lower() not in map(str.lower, preprocessed_dataset[text]["relations"]):
                preprocessed_dataset[text]["relations"].append(relation)
                preprocessed_dataset[text]["offsets"].append(offset)
        else:
            preprocessed_dataset[text] = {"relations": [relation], "offsets": [offset]}

    # Step 2: Sort the relations
    sorted_examples = []
    for text, example in preprocessed_dataset.items():
        relations = util.sort_by_offset(example["relations"], example["offsets"])
        sorted_examples.append(f"{text}\t{' '.join(relations)}")

    return sorted_examples


@app.callback(invoke_without_command=True)
def main(output_dir: Path) -> None:
    """Download and preprocess the ADE V2 corpus for use with seq2rel."""
    msg.divider("Preprocessing ADE")

    with msg.loading("Downloading corpus..."):
        dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")["train"]
    msg.good("Downloaded the corpus.")

    with msg.loading("Preprocessing the data..."):
        preprocessed_dataset = _preprocess(dataset)
        valid_size, test_size = 0.1, 0.2
        train, valid, test = util.train_valid_test_split(
            preprocessed_dataset, valid_size=valid_size, test_size=test_size
        )

    msg.good("Preprocessed the data.")
    msg.info(
        (
            f"Holding out {valid_size:.2%} of the training data as a validation set"
            f" and {test_size:.2%} as a test set."
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}.")


if __name__ == "__main__":
    app()
