from pathlib import Path
from typing import List

import typer
from seq2rel_ds.common import util

app = typer.Typer()

TRAIN_FILENAME = "CDR_TrainingSet.PubTator.txt"
VALID_FILENAME = "CDR_DevelopmentSet.PubTator.txt"
TEST_FILENAME = "CDR_TestSet.PubTator.txt"


def _preprocess_bc5cdr(filepath: Path) -> List[str]:
    pubtator_content = Path(filepath).read_text()
    parsed = util.parse_pubtator(
        pubtator_content=pubtator_content, text_segment=util.TextSegment.both
    )

    processed_dataset = []

    for annotation in parsed.values():
        relations = []
        offsets = []

        for rel in annotation.relations:
            uid_1, uid_2, rel_label = rel
            # Keep track of the end offsets of each entity. We will use these to sort
            # relations according to their order of first appearence in the text.
            offset_1 = min((end for _, end in annotation.clusters[uid_1].offsets))
            offset_2 = min((end for _, end in annotation.clusters[uid_2].offsets))
            offset = offset_1 + offset_2
            ent_clusters = [annotation.clusters[uid_1].ents, annotation.clusters[uid_2].ents]
            ent_labels = [annotation.clusters[uid_1].label, annotation.clusters[uid_2].label]
            relation = util.format_relation(
                ent_clusters=ent_clusters,
                ent_labels=ent_labels,
                rel_label=rel_label,
            )
            relations.append(relation)
            offsets.append(offset)

        relations = util.sort_by_offset(relations, offsets)
        processed_dataset.append(f"{annotation.text}\t{' '.join(relations)}")

    return processed_dataset


@app.callback(invoke_without_command=True)
def main(
    input_dir: Path = typer.Argument(..., help="Path to a local copy of the BC5CDR corpus."),
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
) -> None:
    """Given a path to a local copy of the BC5CDR corpus (`input_dir`), saves a preprocessed copy
    of the data to `output_dir` that can be used to train a model with the seq2rel package.

    See https://github.com/JohnGiorgi/seq2rel for more details on the seq2rel package and
    https://pubmed.ncbi.nlm.nih.gov/27161011/ for more details on the BC5CDR corpus.
    """
    train_filepath = Path(input_dir) / TRAIN_FILENAME
    dev_filepath = Path(input_dir) / VALID_FILENAME
    test_filepath = Path(input_dir) / TEST_FILENAME

    train = _preprocess_bc5cdr(train_filepath)
    valid = _preprocess_bc5cdr(dev_filepath)
    test = _preprocess_bc5cdr(test_filepath)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))


if __name__ == "__main__":
    app()
