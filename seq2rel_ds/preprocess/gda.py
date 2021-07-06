from pathlib import Path
from typing import Dict, List, Tuple

import requests
import typer
from seq2rel_ds import msg
from seq2rel_ds.common import util
from seq2rel_ds.preprocess.util import EntityHinting
from sklearn.model_selection import train_test_split

app = typer.Typer()

GDA_DATA_URL = (
    "https://bitbucket.org/alexwuhkucs/gda-extraction/raw/fd4a7409365e5ff35f9ac9a9fee6755bd34465cd"
)
TRAIN_DATA = "training_data"
TEST_DATA = "testing_data"
ABSTRACTS_FILENAME = "abstracts.txt"
ANNS_FILENAME = "anns.txt"
LABELS_FILENAME = "labels.csv"
REL_LABEL = "GDA"
VALID_SIZE = 0.20


def _download_corpus() -> Tuple[List[List[str]], List[List[str]]]:
    train_abstracts = requests.get(GDA_DATA_URL + f"/{TRAIN_DATA}/" + ABSTRACTS_FILENAME).text
    train_anns = requests.get(GDA_DATA_URL + f"/{TRAIN_DATA}/" + ANNS_FILENAME).text
    train_labels = requests.get(GDA_DATA_URL + f"/{TRAIN_DATA}/" + LABELS_FILENAME).text

    test_abstracts = requests.get(GDA_DATA_URL + f"/{TEST_DATA}/" + ABSTRACTS_FILENAME).text
    test_anns = requests.get(GDA_DATA_URL + f"/{TEST_DATA}/" + ANNS_FILENAME).text
    test_labels = requests.get(GDA_DATA_URL + f"/{TEST_DATA}/" + LABELS_FILENAME).text

    train = [train_abstracts, train_anns, train_labels]
    test = [test_abstracts, test_anns, test_labels]

    return train, test


def _parse_abstracts(abstracts: str) -> Dict[str, Dict[str, str]]:
    parsed_abstracts = {}
    for article in abstracts.strip().split("\n\n"):
        # The article may or may not contain an abstract
        article_split = article.strip().split("\n")
        pmid, title = article_split[:2]
        abstract = article_split[2] if len(article_split) == 3 else ""
        title = util.sanitize_text(title)
        abstract = util.sanitize_text(abstract)
        parsed_abstracts[pmid] = {"title": title, "abstract": abstract}
    return parsed_abstracts


def _parse_labels(labels: str) -> Dict[str, List[str]]:
    parsed_labels: Dict[str, List[str]] = {}
    # First line is a header
    for label in labels.strip().split("\n")[1:]:
        pmid, gene_id, disease_id, _ = label.strip().split(",")
        pubtator_formatted_label = f"{pmid}\t{REL_LABEL}\t{gene_id}\t{disease_id}"
        if pmid in parsed_labels:
            parsed_labels[pmid].append(pubtator_formatted_label)
        else:
            parsed_labels[pmid] = [pubtator_formatted_label]
    return parsed_labels


def _convert_to_pubtator(abstracts: str, anns: str, labels: str) -> str:
    parsed_abstracts = _parse_abstracts(abstracts)
    parsed_labels = _parse_labels(labels)

    pubtator_formatted_anns = []
    for ann in anns.strip().split("\n\n"):
        pmid = ann.strip().split("\n")[0].split("\t")[0].strip()
        pubtator_formatted_title = f"{pmid}|t|{parsed_abstracts[pmid]['title']}"
        pubtator_formatted_abstract = f"{pmid}|a|{parsed_abstracts[pmid]['abstract']}"
        pubtator_formatted_ann = "\n".join(
            [
                pubtator_formatted_title,
                pubtator_formatted_abstract,
                ann.strip(),
                "\n".join(parsed_labels[pmid]),
            ]
        )
        pubtator_formatted_anns.append(pubtator_formatted_ann)

    return "\n\n".join(pubtator_formatted_anns)


def _preprocess(
    abstracts: str, anns: str, labels: str, sort_rels: bool = True, include_ent_hints: bool = False
) -> List[str]:

    pubtator_content = _convert_to_pubtator(abstracts=abstracts, anns=anns, labels=labels)
    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content, text_segment=util.TextSegment.both, sort_ents=True
    )

    seq2rel_annotations = util.pubtator_to_seq2rel(
        pubtator_annotations, sort_rels=sort_rels, include_ent_hints=include_ent_hints
    )

    return seq2rel_annotations


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Directory path to save the preprocessed data."),
    sort_rels: bool = typer.Option(
        True, help="Sort relations according to order of first appearance."
    ),
    entity_hinting: EntityHinting = typer.Option(
        EntityHinting.none,
        help=(
            'Entity hinting strategy. Pass "gold" to use the gold standard annotations, "pipeline"'
            ' to use annotations predicted by a pretrained model, and "none" to not include entity hints.'
        ),
        case_sensitive=False,
    ),
) -> None:
    """Download and preprocess the GDA corpus for use with seq2rel."""
    msg.divider("Preprocessing GDA")

    with msg.loading("Downloading corpus..."):
        train_raw, test_raw = _download_corpus()
    msg.good("Downloaded the corpus")

    include_ent_hints = False
    if entity_hinting == EntityHinting.pipeline:
        raise NotImplementedError(
            "pipeline entity hinting is not implemented for the GDA corpus."
            ' Please use "gold" or "none"'
        )
    elif entity_hinting == EntityHinting.gold:
        include_ent_hints = True
        msg.info("Entity hints will be inserted into the source text using the gold annotations.")

    with msg.loading("Preprocessing the training data..."):
        train = _preprocess(*train_raw, sort_rels=sort_rels, include_ent_hints=include_ent_hints)
    msg.good("Preprocessed the training data")
    with msg.loading("Preprocessing the test data..."):
        test = _preprocess(*test_raw, sort_rels=sort_rels, include_ent_hints=include_ent_hints)
    msg.good("Preprocessed the test data")

    train, valid = train_test_split(train, test_size=VALID_SIZE)
    msg.info(f"Holding out {VALID_SIZE:.2%} of the training data as a validation set")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    app()
