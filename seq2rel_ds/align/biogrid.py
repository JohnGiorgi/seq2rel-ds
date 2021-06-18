import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from seq2rel_ds import msg
from seq2rel_ds.align.util import query_pubtator
from seq2rel_ds.common.schemas import AlignedExample, PydanticEncoder, as_pubtator_annotation
from seq2rel_ds.common.util import TextSegment, pubtator_to_seq2rel, sanitize_text, sort_by_offset

app = typer.Typer()


ENTREZGENE_INTERACTOR_A = "Entrez Gene Interactor A"
ENTREZGENE_INTERACTOR_B = "Entrez Gene Interactor B"
EXPERIMENTAL_SYSTEM_TYPE = "Experimental System Type"
PUBLICATION_SOURCE = "Publication Source"
ORGANISM_ID_INTERACTOR_A = "Organism ID Interactor A"
ORGANISM_ID_INTERACTOR_B = "Organism ID Interactor B"
ORGANISM_NAME_INTERACTOR_A = "Organism Name Interactor A"
ORGANISM_NAME_INTERACTOR_B = "Organism Name Interactor B"
BIOGRID_COLS = {
    ENTREZGENE_INTERACTOR_A: pd.StringDtype(),
    ENTREZGENE_INTERACTOR_B: pd.StringDtype(),
    EXPERIMENTAL_SYSTEM_TYPE: pd.StringDtype(),
    PUBLICATION_SOURCE: pd.StringDtype(),
    ORGANISM_ID_INTERACTOR_A: pd.StringDtype(),
    ORGANISM_ID_INTERACTOR_B: pd.StringDtype(),
    ORGANISM_NAME_INTERACTOR_A: pd.StringDtype(),
    ORGANISM_NAME_INTERACTOR_B: pd.StringDtype(),
}

# Any hardcoded filenames should go here
ANNOTATIONS_FN = "annotations.json"
BIOGRID_FN = "biogrid.tsv"

# Any hardcoded entity or relation labels should go here
REL_LABEL = "GGP"


def _load_biogrid(biogrid_path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(biogrid_path_or_url, sep="\t", usecols=BIOGRID_COLS.keys(), dtype=BIOGRID_COLS)
    return df


def _align(
    input_dir: Path,
    output_fp: Optional[Path] = None,
    max_instances: Optional[int] = None,
    pmid_whitelist: Optional[List[str]] = None,
    sort_rels: bool = True,
    include_ent_hints: bool = False,
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
        for pmid in pmids:
            annotation = annotations[pmid]
            rows = df[df[PUBLICATION_SOURCE] == f"PUBMED:{pmid}"]
            for _, row in rows.iterrows():
                rel_label = row[EXPERIMENTAL_SYSTEM_TYPE]
                interactor_a, interactor_b = (
                    row[ENTREZGENE_INTERACTOR_A],
                    row[ENTREZGENE_INTERACTOR_B],
                )
                cluster_a = annotation.clusters.get(interactor_a, [])
                cluster_b = annotation.clusters.get(interactor_b, [])
                # If we have a hit for both interactors, accumulate the annotation
                if cluster_a and cluster_b:
                    offset_a = min(end for _, end in cluster_a.offsets)
                    offset_b = min(end for _, end in cluster_b.offsets)
                    interactor_a, interactor_b = sort_by_offset(
                        [interactor_a, interactor_b], [offset_a, offset_b]
                    )
                    relation = (interactor_a, interactor_b, rel_label)
                    if relation not in annotation.relations:
                        annotation.relations.append(relation)

            if annotation.relations:
                # Score the alignment as the fraction of BioGRID interactions we found
                score = len(annotation.relations) / len(rows)
                text, relations = pubtator_to_seq2rel(
                    {pmid: annotation}, sort_rels=sort_rels, include_ent_hints=include_ent_hints
                )[0].split("\t")

                aligned_example = AlignedExample(
                    doc_id=pmid, text=text, relations=relations, score=score
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
        TextSegment.both, help="Whether to use title text, abstract text, or both"
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
    # Remove duplicates as these cause problems downstream. Often, these are the same
    # interaction identified by different experimental systems, or the same interactors
    # with opposite directionality.
    # See https://wiki.thebiogrid.org/doku.php/statistics for more details.
    df = df.drop_duplicates()
    msg.info(f"Dropped {num_records - len(df)} ({1 - len(df)/num_records:.2%}) duplicates")
    # Get all unique PMIDs in the dataframe.
    pmids = df[PUBLICATION_SOURCE].apply(lambda x: re.search(r"\d+$", x).group()).unique().tolist()
    msg.info(f"Found {len(pmids)} unique PMIDs in the BioGRID release")

    # Get the PubTator results for all PMIDs in the dataframe
    with msg.loading("Fetching PubTator annotations..."):
        annotations = query_pubtator(
            pmids,
            concepts=["gene"],
            text_segment=text_segment,
            sort_ents=True,
            skip_malformed=True,
        )
    msg.good(f"Fetched {len(annotations)} PubTator annotations")

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
    sort_rels: bool = typer.Option(True),
    include_ent_hints: bool = typer.Option(False),
) -> None:
    """Use BioGRID to create data for distantly supervised learning with seq2rel."""

    _align(
        input_dir=input_dir,
        output_fp=output_dir,
        max_instances=max_instances,
        sort_rels=sort_rels,
        include_ent_hints=include_ent_hints,
    )


if __name__ == "__main__":
    app()
