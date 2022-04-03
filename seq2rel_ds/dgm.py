from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from boltons import jsonutils
from sklearn.model_selection import train_test_split

from seq2rel_ds import msg
from seq2rel_ds.common import text_utils, util
from seq2rel_ds.common.util import EntityHinting

app = typer.Typer()


def _convert_to_pubtator(examples: List[Dict[str, Any]]) -> str:
    """Converts data from the DGM corpus to a PubTator Annotation-like format."""
    pubtator_formatted_anns = []
    for example in examples:
        pmid = example["pmid"]
        paragraphs = example["paragraphs"]
        abstract = " ".join([" ".join(paragraph) for paragraph in example["paragraphs"]])
        # The text contains \t and \n, which will cause issues downstream. Replace them with spaces.
        abstract = text_utils.sanitize_text(abstract)

        # Convert the mentions to PubTator format.
        pubtator_mentions = ""
        for paragraph, mentions in zip(paragraphs, example["mentions"]):
            for mention in mentions:
                start, end, ent_type, uid = (
                    mention["start"],
                    mention["end"],
                    mention["type"],
                    mention["name"],
                )
                # The corpus gives us token offsets, but we need character offsets. To handle this,
                # we join the mention tokens on whitespace and clean the resulting string similar
                # to the abstract text, and then find and accumulate all overlapping mentions in
                # the cleaned abstract tex.
                mention_text = " ".join(paragraph[start:end]).strip()
                mention_text = text_utils.sanitize_text(mention_text)
                char_offsets = [
                    (start, start + len(mention_text))
                    for start in text_utils.findall(abstract, mention_text)
                ]
                for char_start, char_end in char_offsets:
                    pubtator_mentions += (
                        f"{pmid}\t{char_start}\t{char_end}\t{mention_text}\t{ent_type}\t{uid}\n"
                    )
        pubtator_mentions = pubtator_mentions.strip()

        # Convert the relations to PubTator format.
        pubtator_rels = ""
        for candidate in example["triple_candidates"]:
            if candidate["label"] == 0:
                continue
            pubtator_rels += (
                f"{pmid}\tDGM\t{candidate['drug']}\t{candidate['gene']}\t{candidate['variant']}\n"
            )
        pubtator_rels = pubtator_rels.strip()

        # Accumulate the final, PubTator-formatted annotation.
        pubtator = f"{pmid}|t|\n{pmid}|a|{abstract}\n{pubtator_mentions}\n{pubtator_rels}"
        pubtator_formatted_anns.append(pubtator)

    return "\n\n".join(pubtator_formatted_anns)


def _preprocess(
    examples: List[Dict[str, Any]],
    sort_rels: bool = True,
    entity_hinting: Optional[EntityHinting] = None,
) -> List[str]:
    kwargs = (
        {"concepts": ["chemical", "gene", "mutation"], "skip_malformed": True}
        if entity_hinting
        else {}
    )

    pubtator_content = _convert_to_pubtator(examples)
    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content,
        text_segment=util.TextSegment.abstract,
    )
    seq2rel_annotations = util.pubtator_to_seq2rel(
        pubtator_annotations, sort_rels=sort_rels, entity_hinting=entity_hinting, **kwargs
    )
    return seq2rel_annotations


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="Path to a local copy of the DGM corpus."),
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
    valid_size: float = typer.Option(
        0.2,
        help=(
            "Fraction of training examples to hold out as a validation set. The original validation"
            " set is used as the test set, as the test set does not contain paragraph-level annotations"
        ),
    ),
) -> None:
    """Download and preprocess the DGM corpus for use with seq2rel.
    The corpus can be downloaded at: https://hanover.azurewebsites.net/downloads/naacl2019.aspx.
    Provide this path as argument `input_dir` to this command. More details about the corpus can be
    found here: https://arxiv.org/abs/1904.02347. Note that we use the paragraph-length text.
    Because there are no annotations on the paragraph-level, we use the original validation set as
    the test set, and hold out `valid_size` fraction of the training set as the validation set.
    """
    msg.divider("Preprocessing DGM")

    data_fp = Path(input_dir) / "data"
    documents_fp = data_fp / "examples_v2" / "paragraph"
    train_fp = documents_fp / "ds_train_dev.txt"

    if entity_hinting == EntityHinting.pipeline:
        msg.info(
            "Entity hints will be inserted into the source text using the annotations from PubTator."
        )
    elif entity_hinting == EntityHinting.gold:
        msg.info("Entity hints will be inserted into the source text using the gold annotations.")

    # Load in the raw distant supervision data. We follow https://arxiv.org/abs/1904.02347 by
    # removing examples that don't contain triple candidates.
    distsup_raw = [
        line for line in jsonutils.JSONLIterator(open(train_fp)) if line["triple_candidates"]
    ]

    # Split it into train/validation sets.
    distsup_train_pmids = (
        (data_fp / "distsup_pmid_split" / "train.txt").read_text().strip().splitlines()
    )
    distsup_dev_pmids = (
        (data_fp / "distsup_pmid_split" / "dev.txt").read_text().strip().splitlines()
    )
    train_raw = [example for example in distsup_raw if example["pmid"] in distsup_train_pmids]
    valid_raw = [example for example in distsup_raw if example["pmid"] in distsup_dev_pmids]

    with msg.loading("Preprocessing the data..."):
        train = _preprocess(train_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
        # Because we don't have paragraph level annotations for the test set, we will use
        # the entire validation set as the test set and hold out some data from the training set
        # as the new validation set.
        test = _preprocess(valid_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
    msg.good("Preprocessed the data.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if valid_size and valid_size > 0:
        msg.info(f"Holding out {valid_size:.2%} of the training data as a validation set.")
        train, valid = train_test_split(train, test_size=valid_size)
        (output_dir / "valid.tsv").write_text("\n".join(valid))

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}.")


if __name__ == "__main__":
    app()
