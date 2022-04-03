import itertools
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import typer

from seq2rel_ds import msg
from seq2rel_ds.common import util
from seq2rel_ds.common.schemas import PubtatorAnnotation
from seq2rel_ds.common.util import EntityHinting

app = typer.Typer()

CDR_URL = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"
MESH_TREE_URL = (
    "https://github.com/fenchri/edge-oriented-graph/raw/master/data_processing/2017MeshTree.txt"
)
PARENT_DIR = "CDR_Data/CDR.Corpus.v010516"
TRAIN_FILENAME = "CDR_TrainingSet.PubTator.txt"
VALID_FILENAME = "CDR_DevelopmentSet.PubTator.txt"
TEST_FILENAME = "CDR_TestSet.PubTator.txt"


@lru_cache()
def _download_mesh_tree() -> Dict[str, List[str]]:
    """Downloads the MeSH tree and returns a dictionary mapping MeSH unique IDs to tree numbers."""
    parsed_mesh_tree = defaultdict(list)
    raw_mesh_tree = requests.get(MESH_TREE_URL).text.strip().splitlines()[1:]
    for line in raw_mesh_tree:
        tree_numbers, mesh_unique_id, _ = line.split("\t")
        parsed_mesh_tree[mesh_unique_id].append(tree_numbers)
    return parsed_mesh_tree


def _download_corpus() -> Tuple[str, str, str]:
    z = util.download_zip(CDR_URL)
    train = z.read(str(Path(PARENT_DIR) / TRAIN_FILENAME)).decode()
    valid = z.read(str(Path(PARENT_DIR) / VALID_FILENAME)).decode()
    test = z.read(str(Path(PARENT_DIR) / TEST_FILENAME)).decode()

    return train, valid, test


def _filter_hypernyms(pubtator_annotations: List[PubtatorAnnotation]) -> None:
    """For each document in `pubtator_annotations`, determines any possible negative relations
    which are hypernyms of the positive relations. If found, these are appended to
    `pubtator_annotations.filtered_relations`.
    """
    # Download the MeSH tree which allows us to determine hypernyms for disease entities.
    mesh_tree = _download_mesh_tree()

    # Determine the entity and relation labels by looping until we find an document with relations.
    for annotation in pubtator_annotations:
        if annotation.relations:
            chem_id, diso_id, rel_label = annotation.relations[0]
            chem_label = annotation.clusters[chem_id].label
            diso_label = annotation.clusters[diso_id].label
            break

    for annotation in pubtator_annotations:
        # We will add this attribute to each annotation, regardless of whether or not it has
        # relations to filter. This will mean that all examples in the dataset will be formatted
        # the same way, which simplifies data loading.
        annotation.filtered_relations = []
        # Determine the negative relations by taking the set of the product of all unique chemical
        # and disease entities, minus the set of all positive relations.
        chemicals = [
            ent_id for ent_id, ann in annotation.clusters.items() if ann.label == chem_label
        ]
        diseases = [
            ent_id for ent_id, ann in annotation.clusters.items() if ann.label == diso_label
        ]
        all_relations = [
            (chem, diso, rel_label) for chem, diso in itertools.product(chemicals, diseases)
        ]
        negative_relations = list(set(all_relations) - set(annotation.relations))
        # If any negative relation contains a chemical entity that matches the chemical entity of
        # a positive relation AND its disease entity is a hypernym of the positive relations disease
        # entity, this negative relation should be filtered.
        for neg_chem, neg_diso, _ in negative_relations:
            for pos_chem, pos_diso, _ in annotation.relations:
                if neg_chem == pos_chem:
                    if any(
                        neg_tree_number in pos_tree_number
                        for pos_tree_number in mesh_tree[pos_diso]
                        for neg_tree_number in mesh_tree[neg_diso]
                    ):
                        filtered_rel = (neg_chem, neg_diso, rel_label)
                        if filtered_rel not in annotation.filtered_relations:
                            annotation.filtered_relations.append(filtered_rel)


def _preprocess(
    pubtator_content: str,
    sort_rels: bool = True,
    entity_hinting: Optional[EntityHinting] = None,
    filter_hypernyms: bool = False,
) -> List[str]:
    kwargs = {"concepts": ["chemical", "disease"], "skip_malformed": True} if entity_hinting else {}

    pubtator_annotations = util.parse_pubtator(
        pubtator_content=pubtator_content,
        text_segment=util.TextSegment.both,
    )

    # This is unique the the CDR corpus, which contains many negative relations that are
    # actually valid, but are not annotated because they contain a disease entity which is the
    # hypernym of a disease entity in a positive relation. We need to filter these out before
    # evaluation, so this function finds all such cases and adds them to the filtered_relations
    # field of the annoations. See: https://arxiv.org/abs/1909.00228 for details.
    if filter_hypernyms:
        _filter_hypernyms(pubtator_annotations)

    seq2rel_annotations = util.pubtator_to_seq2rel(
        pubtator_annotations,
        sort_rels=sort_rels,
        entity_hinting=entity_hinting,
        **kwargs,
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
    combine_train_valid: bool = typer.Option(
        False, help="Combine the train and validation sets into one train set."
    ),
) -> None:
    """Download and preprocess the CDR corpus for use with seq2rel."""
    msg.divider("Preprocessing CDR")

    with msg.loading("Downloading corpus..."):
        train_raw, valid_raw, test_raw = _download_corpus()
    msg.good("Downloaded the corpus.")

    if entity_hinting == EntityHinting.pipeline:
        msg.info(
            "Entity hints will be inserted into the source text using the annotations from PubTator."
        )
    elif entity_hinting == EntityHinting.gold:
        msg.info("Entity hints will be inserted into the source text using the gold annotations.")

    with msg.loading("Preprocessing the data..."):
        if combine_train_valid:
            msg.info("Training and validation sets will be combined into one train set.")
            train_raw = f"{train_raw.strip()}\n\n{valid_raw.strip()}"
            valid = None
        else:
            valid = _preprocess(
                valid_raw, sort_rels=sort_rels, entity_hinting=entity_hinting, filter_hypernyms=True
            )
        train = _preprocess(train_raw, sort_rels=sort_rels, entity_hinting=entity_hinting)
        test = _preprocess(
            test_raw, sort_rels=sort_rels, entity_hinting=entity_hinting, filter_hypernyms=True
        )
    msg.good("Preprocessed the data.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    if valid:
        (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}.")


if __name__ == "__main__":
    app()
