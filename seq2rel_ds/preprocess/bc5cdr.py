import io
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import typer
from seq2rel_ds import msg
from seq2rel_ds.common import util
from seq2rel_ds.common.util import EntityHinting

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


def _preprocess(
    pubtator_content: str,
    sort_rels: bool = True,
    entity_hinting: Optional[EntityHinting] = None,
) -> List[str]:

    kwargs = {"concepts": ["chemical", "disease"], "skip_malformed": True} if entity_hinting else {}

    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content, text_segment=util.TextSegment.both, sort_ents=True
    )
    seq2rel_annotations = util.pubtator_to_seq2rel(
        pubtator_annotations, sort_rels=sort_rels, entity_hinting=entity_hinting, **kwargs
    )

    return seq2rel_annotations


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
    sort_rels: bool = typer.Option(
        True, help="Sort relations according to order of first appearance."
    ),
    entity_hinting: EntityHinting = typer.Option(
        None,
        help=(
            'Entity hinting strategy. Pass "gold" to use the gold standard annotations, "pipeline"'
            " to use annotations predicted by a pretrained model, or omit it to not include entity hints."
        ),
        case_sensitive=False,
    ),
) -> None:
    """Download and preprocess the BC5CDR corpus for use with seq2rel."""
    msg.divider("Preprocessing BC5CDR")

    with msg.loading("Downloading corpus..."):
        train_raw, valid_raw, test_raw = _download_corpus()
    msg.good("Downloaded the corpus")

    if entity_hinting == EntityHinting.pipeline:
        msg.info(
            "Entity hints will be inserted into the source text using the annotations from PubTator."
        )
    elif entity_hinting == EntityHinting.gold:
        msg.info("Entity hints will be inserted into the source text using the gold annotations.")

    with msg.loading("Preprocessing the data..."):
        train = _preprocess(train_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
        valid = _preprocess(valid_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
        test = _preprocess(test_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
    msg.good("Preprocessed the data")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    app()
