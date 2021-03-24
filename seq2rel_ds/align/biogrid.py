import json
import re
from math import ceil
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
import typer
from fuzzywuzzy import fuzz
from joblib import Memory
from more_itertools import chunked
from enum import Enum

from scispacy.abbreviation import AbbreviationDetector  # noqa
from seq2rel_ds import msg
from seq2rel_ds.common.util import (
    fuzzy_match,
    get_pubtator_response,
    get_uniprot_synonyms,
    sanitize_text,
    set_seeds,
)
from spacy.tokens.doc import Doc
from itertools import chain

app = typer.Typer(callback=set_seeds)


# List of problems:
# - Multiple examples per pmid (should be one)
# - Sorting by order of first apperance
# - Make a test set
# - Relations with the same entities
# - Duplicates
# - unclosed paranthases
# - entities aren't marked here!
# - include pubtator relations?
# - use pubtator for retrieving text as well?

INTERACTOR_A_SYMBOL = "Official Symbol Interactor A"
INTERACTOR_B_SYMBOL = "Official Symbol Interactor B"
INTERACTOR_A_SYNONYMS = "Synonyms Interactor A"
INTERACTOR_B_SYNONYMS = "Synonyms Interactor B"
EXPERIMENTAL_SYSTEM_TYPE = "Experimental System Type"
PUBLICATION_SOURCE = "Publication Source"
INTERACTOR_A_UNIPROT_ID = "SWISS-PROT Accessions Interactor A"
INTERACTOR_B_UNIPROT_ID = "SWISS-PROT Accessions Interactor B"
BIOGRID_COLS = [
    INTERACTOR_A_SYMBOL,
    INTERACTOR_B_SYMBOL,
    INTERACTOR_A_SYNONYMS,
    INTERACTOR_B_SYNONYMS,
    EXPERIMENTAL_SYSTEM_TYPE,
    PUBLICATION_SOURCE,
    INTERACTOR_A_UNIPROT_ID,
    INTERACTOR_B_UNIPROT_ID,
]
END_OF_REL_SYMBOL = "@EOR@"
SYNONYMS_FN = "synonyms.json"
ANNOTATIONS_FN = "annotations.json"
BIOGRID_FN = "biogrid.tsv"


class TextSegment(str, Enum):
    title = "title"
    abstract = "abstract"
    both = "both"


# TODO: Would be better to move this to an __init__ so all files in the directory can use it.
# Setup caching
CACHE_DIR = Path.home() / ".seq2rel_ds"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)
    msg.info(f"Created a cache directory at: {CACHE_DIR}")
else:
    msg.info(f"Using cache directory at: {CACHE_DIR}")
memory = Memory(CACHE_DIR, verbose=0)


def _load_scipacy(model: str):
    # We don't need the tagger or parser.
    # Disabling them will speed up processing.
    nlp = spacy.load(model, disable=["tagger", "parser"])
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
    return nlp


def _load_biogrid(biogrid_path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(biogrid_path_or_url, sep="\t", usecols=BIOGRID_COLS)
    return df


def _format_rel(interactor_a: str, interactor_b: str, rel_type: str) -> str:
    return f"@{rel_type.strip().upper()}@ {interactor_a} @PRGE@ {interactor_b} @PRGE@ {END_OF_REL_SYMBOL}"


def _get_entities(doc: Doc) -> List[str]:
    # Use spaCy to extract entities
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    # Attempt to include full mentions of an entitiy, including its abbreviation.
    try:
        for abrv in doc._.abbreviations:
            entities.append(f"{abrv._.long_form.text} ({abrv.text})")
    except AttributeError:
        pass
    return entities


def _sort_by_offset(items: List[str], offsets: List[int], **kwargs) -> List[str]:
    packed = list(zip(items, offsets))
    packed = sorted(packed, key=itemgetter(1), **kwargs)
    items, offsets = list(zip(*packed))
    return items


def _align_row(row, text: str, ents: List[str], synonyms: Dict[str, List[str]]) -> Tuple[str]:
    # Get all the information we need from BioGrid df
    interactor_a = [row[INTERACTOR_A_SYMBOL]] + row[INTERACTOR_A_SYNONYMS].split("|")
    interactor_b = [row[INTERACTOR_B_SYMBOL]] + row[INTERACTOR_B_SYNONYMS].split("|")
    uniprot_id_a = row[INTERACTOR_A_UNIPROT_ID].split("|")
    uniprot_id_b = row[INTERACTOR_B_UNIPROT_ID].split("|")

    # Expand the list of interactor names with their synonyms
    interactor_a_synonyms = chain.from_iterable(synonyms[_id] for _id in uniprot_id_a)
    interactor_b_synonyms = chain.from_iterable(synonyms[_id] for _id in uniprot_id_b)
    interactor_a.extend(interactor_a_synonyms)
    interactor_b.extend(interactor_b_synonyms)

    # Remove duplicates (order doesn't matter)
    interactor_a = list(set(interactor_a))
    interactor_b = list(set(interactor_b))

    # Fuzzy match the extracted entities to the interactor names
    interactor_a_match = fuzzy_match(
        interactor_a, ents, scorer=fuzz.token_sort_ratio, score_cutoff=90, limit=3
    )
    interactor_b_match = fuzzy_match(
        interactor_b, ents, scorer=fuzz.token_sort_ratio, score_cutoff=90, limit=3
    )

    # TODO: Here, we make a strong assumption that only dimers will contain interactions
    # between identical entities. There is almost definitely a better way to handle this.
    if interactor_a_match == interactor_b_match and "dimer" not in text:
        interactor_a_match, interactor_b_match = "", ""

    return interactor_a_match, interactor_b_match


def _align(
    input_dir: str,
    output_fp: Optional[str] = None,
    text_segment: TextSegment = TextSegment.both,
    max_instances: Optional[int] = None,
    pmid_whitelist: Optional[List[str]] = None,
) -> List[str]:
    msg.divider("Alignment")

    # TODO: Should check if input_dir exists, raising an error if not. Can we do that
    # in the typer Argument declaration.
    input_dir = Path(input_dir)
    if output_fp:
        output_fp = Path(output_fp)

    synonyms_fp = input_dir / SYNONYMS_FN
    annotations_fp = input_dir / ANNOTATIONS_FN
    biogrid_fp = input_dir / BIOGRID_FN

    synonyms = json.loads(synonyms_fp.read_text())
    msg.good(f"Loaded synonyms file at: {synonyms_fp}")
    annotations = json.loads(annotations_fp.read_text())
    msg.good(f"Loaded annotations file at: {annotations_fp}")
    df = pd.read_csv(biogrid_fp, sep="\t")
    msg.good(f"Loaded BioGRID file at: {biogrid_fp}")

    if pmid_whitelist:
        pmids = list(set(annotations.keys()) & set(pmid_whitelist))
    else:
        pmids = list(annotations.keys())

    alignments = {}
    max_instances = max_instances or len(pmids)
    with typer.progressbar(length=max_instances, label="Aligning") as progress:
        for i, pmid in enumerate(pmids):
            if text_segment.value == "title":
                text = annotations[pmid]["title"]["text"]
                ents = annotations[pmid]["title"]["ents"]
            elif text_segment.value == "abstract":
                text = annotations[pmid]["abstract"]["text"]
                ents = annotations[pmid]["abstract"]["ents"]
            else:
                text = annotations[pmid]["title"]["text"] + annotations[pmid]["abstract"]["text"]
                ents = annotations[pmid]["title"]["ents"] + annotations[pmid]["abstract"]["ents"]

            rows = df[df[PUBLICATION_SOURCE] == f"PUBMED:{pmid}"]

            relations = []
            offsets = []
            for _, row in rows.iterrows():
                # TODO: In the future, it might be nice to try and distinguish between genetic and
                # physical interactions. For now, we group them into one category.
                # rel_type = row[EXPERIMENTAL_SYSTEM_TYPE]
                rel_type = "interaction"
                interactor_a, interactor_b = _align_row(row, text, ents, synonyms)
                # If we have a hit for both interactors, accumulate the annotation
                if interactor_a and interactor_b:
                    # Keep track of the end offsets of each entity. We will use these to sort
                    # relations according to their order of first appearence in the text.
                    offset_a = text.find(interactor_a) + len(interactor_a)
                    offset_b = text.find(interactor_b) + len(interactor_b)
                    offset = offset_a + offset_b
                    interactor_a, interactor_b = _sort_by_offset(
                        [interactor_a, interactor_b], [offset_a, offset_b]
                    )
                    relation = _format_rel(interactor_a, interactor_b, rel_type)
                    # Some very basic text preprocessing to standardize things
                    relation = sanitize_text(relation)
                    # Don't accumulate identical relations
                    if relation not in relations:
                        relations.append(relation)
                        offsets.append(offset)

            if relations:
                # Sort the relations by order of first appearence
                relations = _sort_by_offset(relations, offsets)
                relations = " ".join(relations)
                annotation = f"{text}\t{relations}"
                if output_fp is not None:
                    with open(output_fp, "a") as f:
                        f.write(f"{annotation}\n")

            # For testing purposes, we still record the pmid even if no relations were found.
            alignments[pmid] = relations
            progress.update(1)

            if i == max_instances:
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
        if pmid not in alignments:
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
    output_dir=typer.Argument(..., help="Directory to save the resulting data"),
    biogrid_path_or_url: str = typer.Argument(
        ...,
        help="A path on disk (or URL) to a tab delineated (*.tab3) BioGRID release",
    ),
    scispacy_model: str = typer.Option(
        None, help="The scispaCy model to use for NER and abbreviation resolution"
    ),
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msg.divider("Preprocessing")

    if scispacy_model is not None:
        nlp = _load_scipacy(scispacy_model)
        msg.good(f"Loaded scispacy model: {scispacy_model}")
    else:
        nlp = None
        msg.warn(
            "scispacy_model not provided. This will negatively impact the quality of alignments"
        )

    df = _load_biogrid(biogrid_path_or_url)
    msg.good(f"Loaded BioGRID release: {biogrid_path_or_url}")
    # Filter anything without a PMID
    df = df[df[PUBLICATION_SOURCE].str.contains("PUBMED")]
    num_records = len(df)
    # Remove duplicates as these cause problems downstream. Not sure why these exist to begin with.
    df = df.drop_duplicates()
    msg.info(f"Dropped {num_records - len(df)} ({1 - len(df)/num_records:.2%}) duplicates")
    # Get all unique PMIDs in the dataframe.
    pmids = df[PUBLICATION_SOURCE].apply(lambda x: re.search(r"\d+$", x).group()).unique().tolist()
    msg.info(f"Found {len(pmids)} unique PMIDs in the BioGRID release")

    # Get all unique UniProt IDs in the dataframe, account for the compound IDs and duplicates.
    uniprot_ids = df[INTERACTOR_A_UNIPROT_ID].unique().tolist()
    uniprot_ids.extend(df[INTERACTOR_B_UNIPROT_ID].unique().tolist())
    uniprot_ids = [_id for _ids in uniprot_ids for _id in _ids.split("|")]
    uniprot_ids = list(set(uniprot_ids))

    synonyms = {}
    with typer.progressbar(uniprot_ids, label="Fetching UniProt synonyms") as progress:
        for _id in progress:
            synonyms[_id] = get_uniprot_synonyms(_id)

    # Get the PubTator results for all PMIDs in the dataframe
    annotations = {}
    # This is the number of PMIDs per request allowed by pubtators webservice.
    # See: https://www.ncbi.nlm.nih.gov/research/pubtator/api.html for details.
    pmids_per_request = 1000
    total_results = ceil(len(pmids) / pmids_per_request)
    with typer.progressbar(length=total_results, label="Fetching PubTator annotations") as progress:
        for pmids_ in chunked(pmids, pmids_per_request):
            annotation = get_pubtator_response(pmids_, concepts=["gene"])
            annotations.update(annotation)
            progress.update(1)

    # If a ScispaCy model was provided, use it to extend the list of entities.
    if nlp is not None:
        # Batch the inputs for more efficient processing
        batch_size = 100
        num_batches = ceil(len(annotations) / batch_size)
        with typer.progressbar(
            length=num_batches, label="Extending annotations with ScispaCy"
        ) as progress:
            for pmids in chunked(annotations, batch_size):
                # Downstream, we may want to compare performance when aligning to titles,
                # abstracts or both. To keep all options open, we separately keep track of
                # entities in the title and the abstract.
                titles = [annotations[pmid]["title"]["text"] for pmid in pmids]
                abstracts = [annotations[pmid]["abstract"]["text"] for pmid in pmids]

                title_docs = list(nlp.pipe(titles))
                abstract_docs = list(nlp.pipe(abstracts))

                for pmid, title_doc, abstract_doc in zip(pmids, title_docs, abstract_docs):
                    annotations[pmid]["title"]["ents"].extend(_get_entities(title_doc))
                    annotations[pmid]["abstract"]["ents"].extend(_get_entities(abstract_doc))
                progress.update(1)
            # We don't need to retain duplicate entities
            annotations[pmid]["title"]["ents"] = list(set(annotations[pmid]["title"]["ents"]))
            annotations[pmid]["abstract"]["ents"] = list(set(annotations[pmid]["abstract"]["ents"]))

    synonyms_fp = output_dir / SYNONYMS_FN
    annotations_fp = output_dir / ANNOTATIONS_FN
    biogrid_fp = output_dir / BIOGRID_FN

    synonyms_fp.write_text(json.dumps(synonyms, indent=2))
    annotations_fp.write_text(json.dumps(annotations, indent=2))
    df.to_csv(biogrid_fp, sep="\t")


@app.command()
def main(
    input_dir: str = typer.Argument(..., help="Directory containing the preprocessed data"),
    output_dir: str = typer.Argument(..., help="Directory to save the resulting data"),
    text_segment: TextSegment = typer.Option(
        TextSegment.both, help="Whether to use title text, abstract text, or both"
    ),
    max_instances: int = typer.Option(
        ..., help="The maximum number of PMIDs to produce alignments for"
    ),
) -> None:
    """Creates training data for distantly supervised relation extraction by aligning BioGRID
    records to the PubMed articles they were curated from.
    """

    _align(
        input_dir=input_dir,
        output_fp=output_dir,
        text_segment=text_segment,
        max_instances=max_instances,
    )


if __name__ == "__main__":
    app()
