import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
import typer
from seq2rel_ds import msg
from seq2rel_ds.common import util
from seq2rel_ds.common import schemas

app = typer.Typer()

BC5CDR_URL = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"
PARENT_DIR = "CDR_Data/CDR.Corpus.v010516"
TRAIN_FILENAME = "CDR_TrainingSet.PubTator.txt"
VALID_FILENAME = "CDR_DevelopmentSet.PubTator.txt"
TEST_FILENAME = "CDR_TestSet.PubTator.txt"


def _download_corpus() -> Tuple[str, str, str]:
    # https://stackoverflow.com/a/23419450/6578628
    r = requests.get(BC5CDR_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    train = z.read(str(Path(PARENT_DIR) / TRAIN_FILENAME)).decode("utf8")
    valid = z.read(str(Path(PARENT_DIR) / VALID_FILENAME)).decode("utf8")
    test = z.read(str(Path(PARENT_DIR) / TEST_FILENAME)).decode("utf8")

    return train, valid, test


def _preprocess(pubtator_content: str) -> List[str]:
    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content, text_segment=util.TextSegment.both, sort_ents=True
    )
    seq2rel_annotations = util.pubtator_to_seq2rel(pubtator_annotations)

    return seq2rel_annotations


@app.command()
def compute_stats(
    output_dir: Path = typer.Argument(..., help="Directory path to save the corpus statistics")
):
    """Compute corpus statistics for the BC5CDR corpus."""
    msg.divider("Corpus statistics for BC5CDR")

    with msg.loading("Downloading corpus..."):
        train_raw, valid_raw, test_raw = _download_corpus()
    msg.good("Downloaded the corpus")

    with msg.loading("Computing corpus statistics..."):
        import spacy

        nlp = spacy.load("en_core_sci_sm", disable=["ner"])

        train = util.parse_pubtator(
            pubtator_content=train_raw, text_segment=util.TextSegment.both, sort_ents=True
        )
        valid = util.parse_pubtator(
            pubtator_content=valid_raw, text_segment=util.TextSegment.both, sort_ents=True
        )
        test = util.parse_pubtator(
            pubtator_content=test_raw, text_segment=util.TextSegment.both, sort_ents=True
        )

        train_stats = util.compute_corpus_statistics(train, nlp=nlp)
        valid_stats = util.compute_corpus_statistics(valid, nlp=nlp)
        test_stats = util.compute_corpus_statistics(test, nlp=nlp)

    corpus_statistics = schemas.CorpusStatistics(
        train=train_stats, valid=valid_stats, test=test_stats
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fp = output_dir / "corpus_statistics.json"
    with output_fp.open(mode="w") as f:
        json.dump(corpus_statistics.dict(), f, indent=2)
    msg.good(f"Corpus statistics saved to {output_dir.resolve()}")


@app.callback(invoke_without_command=True, context_settings={"ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    output_dir: str = typer.Argument(..., help="Directory path to save the preprocessed data."),
) -> None:
    """Download and preprocess the BC5CDR corpus for use with seq2rel."""
    if ctx.invoked_subcommand is not None:
        return

    msg.divider("Preprocessing BC5CDR")

    with msg.loading("Downloading corpus..."):
        train_raw, valid_raw, test_raw = _download_corpus()
    msg.good("Downloaded the corpus")

    with msg.loading("Preprocessing the data..."):
        train = _preprocess(train_raw)
        valid = _preprocess(valid_raw)
        test = _preprocess(test_raw)
    msg.good("Preprocessed the data")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    app()
