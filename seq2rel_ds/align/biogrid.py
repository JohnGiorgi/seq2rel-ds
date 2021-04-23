import json
import re
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from more_itertools import chunked
from seq2rel_ds import msg
from seq2rel_ds.align.util import query_pubtator
from seq2rel_ds.common.schemas import AlignedExample, PydanticEncoder, as_pubtator_annotation
from seq2rel_ds.common.util import (
    TextSegment,
    format_relation,
    sanitize_text,
    set_seeds,
    sort_by_offset,
)

app = typer.Typer(callback=set_seeds)


ENTREZGENE_INTERACTOR_A = "Entrez Gene Interactor A"
ENTREZGENE_INTERACTOR_B = "Entrez Gene Interactor B"
EXPERIMENTAL_SYSTEM_TYPE = "Experimental System Type"
PUBLICATION_SOURCE = "Publication Source"
BIOGRID_COLS = {
    ENTREZGENE_INTERACTOR_A: pd.StringDtype(),
    ENTREZGENE_INTERACTOR_B: pd.StringDtype(),
    EXPERIMENTAL_SYSTEM_TYPE: pd.StringDtype(),
    PUBLICATION_SOURCE: pd.StringDtype(),
}

# Any hardcoded filenames should go here
ANNOTATIONS_FN = "annotations.json"
BIOGRID_FN = "biogrid.tsv"

# Any hardcoded entity or relation labels should go here
GGP = "GGP"

# This is the number of PMIDs per request allowed by pubtators webservice.
# See: https://www.ncbi.nlm.nih.gov/research/pubtator/api.html for details.
PMIDS_PER_REQUEST = 1000


def _load_biogrid(biogrid_path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(biogrid_path_or_url, sep="\t", usecols=BIOGRID_COLS.keys(), dtype=BIOGRID_COLS)
    return df


def _align(
    input_dir: Path,
    output_fp: Optional[Path] = None,
    text_segment: TextSegment = TextSegment.abstract,
    max_instances: Optional[int] = None,
    pmid_whitelist: Optional[List[str]] = None,
) -> Dict[str, AlignedExample]:
    msg.divider("Alignment")

    # TODO: Should check if input_dir exists, raising an error if not. Can we do that
    # in the typer Argument declaration.
    input_dir = Path(input_dir)
    if output_fp:
        output_fp = Path(output_fp)

    annotations_fp = input_dir / ANNOTATIONS_FN
    biogrid_fp = input_dir / BIOGRID_FN

    annotations = json.loads(annotations_fp.read_text(), object_hook=as_pubtator_annotation)
    msg.good(f"Loaded annotations file at: {annotations_fp}")
    df = pd.read_csv(biogrid_fp, sep="\t", dtype=BIOGRID_COLS)
    msg.good(f"Loaded BioGRID file at: {biogrid_fp}")

    if pmid_whitelist:
        pmids = list(set(annotations.keys()) & set(pmid_whitelist))
    else:
        pmids = list(annotations.keys())
    msg.info(f"Alignments will be generated from {len(pmids)} PMIDs")

    max_instances = max_instances or len(pmids)
    msg.info(f"Total number of alignments will be restricted to {len(pmids)}")

    alignments = {}
    with typer.progressbar(length=max_instances, label="Aligning") as progress:
        for i, pmid in enumerate(pmids):

            rows = df[df[PUBLICATION_SOURCE] == f"PUBMED:{pmid}"]
            annotation = annotations[pmid]

            relations = []
            offsets = []
            for _, row in rows.iterrows():
                rel_label = row[EXPERIMENTAL_SYSTEM_TYPE]
                interactor_a = annotation.clusters.get(row[ENTREZGENE_INTERACTOR_A], [])
                interactor_b = annotation.clusters.get(row[ENTREZGENE_INTERACTOR_B], [])

                # If we have a hit for both interactors, accumulate the annotation
                if interactor_a and interactor_b:
                    # Keep track of end offsets. We will use these to sort entities and relations
                    # according to their order of first appearence in the text.
                    offset_a = min(end for _, end in interactor_a.offsets)
                    offset_b = min(end for _, end in interactor_b.offsets)
                    offset = offset_a + offset_b
                    ent_clusters = sort_by_offset(
                        [interactor_a.ents, interactor_b.ents], [offset_a, offset_b]
                    )
                    ent_labels = [GGP] * len(ent_clusters)
                    relation = format_relation(
                        ent_clusters=ent_clusters, ent_labels=ent_labels, rel_label=rel_label
                    )
                    # Don't accumulate identical relations
                    if relation not in relations:
                        relations.append(relation)
                        offsets.append(offset)

            if relations:
                # Sort the relations by order of first appearence
                relations = sort_by_offset(relations, offsets)
                # Score the alignment as the fraction of BioGRID interactions we found
                score = len(relations) / len(rows)

                aligned_example = AlignedExample(
                    doc_id=pmid, text=annotation.text, relations=" ".join(relations), score=score
                )
                if output_fp is not None:
                    with open(output_fp, "a") as f:
                        f.write(f"{aligned_example.json()}\n")
                progress.update(1)

                # TODO: This leads to empty entries during testing. Come up with a solution.
                alignments[pmid] = aligned_example

            if len(alignments) == max_instances:
                break

    return alignments


@app.command()
def test(
    input_dir=typer.Argument(..., help="Directory containing the preprocessed data"),
    ground_truth_fp: str = typer.Argument(..., help="Path on disk to the ground truth alignments"),
) -> None:
    # load the pmids, text and labels
    ground_truth = {}
    with open(ground_truth_fp, "r") as f:
        for line in f:
            if not line:
                continue
            pmid, _, annotation = line.split("\t")
            # Some very basic preprocessing so we don't fail on things like extra spaces
            annotation = sanitize_text(annotation)
            ground_truth[pmid] = annotation
    msg.good(
        f"Loaded ground truth annnotations at {ground_truth_fp}. Found {len(ground_truth)} PMIDs."
    )

    alignments = _align(
        input_dir=input_dir,
        output_fp="predictions.tsv",
        pmid_whitelist=list(ground_truth.keys()),
    )

    msg.divider("Testing")
    correct, missed = 0, 0
    for pmid, relation in ground_truth.items():
        if not alignments.get(pmid, ""):
            missed += 1
        elif relation == alignments[pmid]:
            correct += 1
    accuracy = correct / len(alignments)
    recall = missed / len(ground_truth)

    msg.text(
        f"Alignments matched the ground truths with {accuracy:.2%} accuracy ({correct}/{len(ground_truth)})"
    )
    if missed > 0:
        msg.warn(f"{missed} ({recall:.2%}) PMIDs in the ground truth were missed during alignment")


@app.command()
def preprocess(
    output_dir: Path = typer.Argument(..., help="Directory to save the resulting data"),
    biogrid_path_or_url: str = typer.Argument(
        ...,
        help="A path on disk (or URL) to a tab delineated (*.tab3) BioGRID release",
    ),
    text_segment: TextSegment = typer.Option(
        TextSegment.abstract, help="Whether to use title text, abstract text, or both"
    ),
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msg.divider("Preprocessing")

    df = _load_biogrid(biogrid_path_or_url)
    msg.good(f"Loaded BioGRID release: {biogrid_path_or_url}")
    # Filter anything without a PMID
    df = df[df[PUBLICATION_SOURCE].str.contains("PUBMED")]
    num_records = len(df)
    # Remove duplicates as these cause problems downstream. Often, these are
    # the same interaction identified by different experimental systems.
    df = df.drop_duplicates()
    msg.info(f"Dropped {num_records - len(df)} ({1 - len(df)/num_records:.2%}) duplicates")
    # Get all unique PMIDs in the dataframe.
    pmids = df[PUBLICATION_SOURCE].apply(lambda x: re.search(r"\d+$", x).group()).unique().tolist()
    msg.info(f"Found {len(pmids)} unique PMIDs in the BioGRID release")

    # Get the PubTator results for all PMIDs in the dataframe
    annotations = {}
    total_results = ceil(len(pmids) / PMIDS_PER_REQUEST)
    with typer.progressbar(length=total_results, label="Fetching PubTator annotations") as progress:
        for pmids_ in chunked(pmids, PMIDS_PER_REQUEST):
            annotation = query_pubtator(
                pmids_, concepts=["gene"], text_segment=text_segment, skip_malformed=True
            )
            annotations.update(annotation)
            progress.update(1)

    annotations_fp = output_dir / ANNOTATIONS_FN
    biogrid_fp = output_dir / BIOGRID_FN

    annotations_fp.write_text(json.dumps(annotations, indent=2, cls=PydanticEncoder))
    msg.good(f"Saved annotations_fp file to: {annotations_fp}")
    df.to_csv(biogrid_fp, sep="\t")
    msg.good(f"Saved preprocessed BioGRID release to: {biogrid_fp}")


@app.command()
def main(
    input_dir: str = typer.Argument(..., help="Directory containing the preprocessed data"),
    output_dir: str = typer.Argument(..., help="Directory to save the resulting data"),
    max_instances: int = typer.Option(
        None, help="The maximum number of PMIDs to produce alignments for."
    ),
) -> None:
    """Creates training data for distantly supervised relation extraction by aligning BioGRID
    records to the PubMed articles they were curated from.
    """

    _align(
        input_dir=input_dir,
        output_fp=output_dir,
        max_instances=max_instances,
    )


if __name__ == "__main__":
    app()
