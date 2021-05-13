import io
import zipfile
from pathlib import Path
from typing import List

import requests
import typer
from seq2rel_ds import msg
from seq2rel_ds.common import util

app = typer.Typer()

BC5CDR_URL = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"
PARENT_DIR = "CDR_Data/CDR.Corpus.v010516"
TRAIN_FILENAME = "CDR_TrainingSet.PubTator.txt"
VALID_FILENAME = "CDR_DevelopmentSet.PubTator.txt"
TEST_FILENAME = "CDR_TestSet.PubTator.txt"


def _preprocess(pubtator_content: str, include_ent_hints: bool = False) -> List[str]:
    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content, text_segment=util.TextSegment.both, sort_ents=True
    )
    seq2rel_annotations = util.pubtator_to_seq2rel(pubtator_annotations, include_ent_hints)

    return seq2rel_annotations


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
    include_ent_hints: bool = typer.Option(
        False, help="Include entity location and type hints in the text"
    ),
) -> None:
    """Preprocess a local copy of the BC5CDR corpus for use with seq2rel."""
    msg.divider("Preprocessing BC5CDR")

    with msg.loading("Downloading corpus..."):
        # https://stackoverflow.com/a/23419450/6578628
        r = requests.get(BC5CDR_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
    msg.good("Downloaded the corpus")

    with msg.loading("Preprocessing the training data..."):
        raw = z.read(str(Path(PARENT_DIR) / TRAIN_FILENAME)).decode("utf8")
        train = _preprocess(raw, include_ent_hints)
    msg.good("Preprocessed the training data")

    with msg.loading("Preprocessing the validation data..."):
        raw = z.read(str(Path(PARENT_DIR) / VALID_FILENAME)).decode("utf8")
        valid = _preprocess(raw, include_ent_hints)
    msg.good("Preprocessed the validation data")

    with msg.loading("Preprocessing the test data..."):
        raw = z.read(str(Path(PARENT_DIR) / TEST_FILENAME)).decode("utf8")
        test = _preprocess(raw, include_ent_hints)
    msg.good("Preprocessed the test data")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    app()
