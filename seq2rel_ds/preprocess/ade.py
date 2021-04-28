from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from datasets import load_dataset
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


def _preprocess() -> List[str]:
    dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")["train"]

    # Step 1: Process the dataset line by line, collecting formatted relations and their offsets.
    processed_dataset: Dict[str, Any] = {}
    with typer.progressbar(dataset, label="Processing") as progress:
        for line in progress:
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
            if text in processed_dataset:
                if relation.lower() not in map(str.lower, processed_dataset[text]["relations"]):
                    processed_dataset[text]["relations"].append(relation)
                    processed_dataset[text]["offsets"].append(offset)
            else:
                processed_dataset[text] = {"relations": [relation], "offsets": [offset]}

    # Step 2: Sort the relations
    sorted_examples = []
    for text, example in processed_dataset.items():
        relations = util.sort_by_offset(example["relations"], example["offsets"])
        sorted_examples.append(f"{text}\t{' '.join(relations)}")

    return sorted_examples


@app.callback(invoke_without_command=True)
def main(output_dir: Path) -> None:
    """Download and preprocess the ADE V2 corpus for use with seq2rel."""
    dataset = _preprocess()
    train, valid, test = util.train_valid_test_split(dataset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))


if __name__ == "__main__":
    app()
