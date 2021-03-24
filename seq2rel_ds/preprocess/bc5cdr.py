from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from seq2rel_ds.common.util import set_seeds

app = typer.Typer(callback=set_seeds)

TRAIN_FILENAME = "CDR_TrainingSet.PubTator.txt"
VALID_FILENAME = "CDR_DevelopmentSet.PubTator.txt"
TEST_FILENAME = "CDR_TestSet.PubTator.txt"


def format_entity(ent_text: str, ent_type: str):
    return f"{ent_text} <{ent_type.upper()}>"


def format_relation(ent_1_text, ent_1_type, ent_2_text, ent_2_type, rel_type):
    ent_1 = format_entity(ent_1_text, ent_1_type)
    ent_2 = format_entity(ent_2_text, ent_2_type)
    return f"<{rel_type.upper()}> {ent_1} {ent_2} </{rel_type.upper()}>"


def parse_entity(annotation: str) -> List[Tuple[str]]:
    """Returns each entity in `annotation` as a list of tuples containing the end index, entity
    text, entity type, and entity ID.
    """
    entities = []
    # This is a simple entity that may have one or more IDs.
    if len(annotation) == 6:
        _, ent_start, _, ent_texts, ent_type, ent_ids = annotation
        ent_ids = ent_ids.split("|")
        ent_texts = [ent_texts] * len(ent_ids)
    # This is a nested entity with multiple IDs.
    else:
        _, ent_start, _, _, ent_type, ent_ids, ent_texts = annotation
        ent_ids = ent_ids.split("|")
        ent_texts = ent_texts.split("|")

    offset = int(ent_start)
    for ent_id, ent_text in zip(ent_ids, ent_texts):
        ent_start, ent_end = offset, offset + len(ent_text)
        entities.append((ent_start, ent_end, ent_text, ent_type, ent_id))

    return entities


def parse_relation(annotation: str, entities: Dict[str, Any]):
    """Formats the relation contained in `annotation`, resolving the IDs using `entities`."""
    _, rel_type, ent_1_id, ent_2_id = annotation
    (
        _,
        ent_1_end,
        ent_1_text,
        ent_1_type,
    ) = entities[ent_1_id]
    _, ent_2_end, ent_2_text, ent_2_type = entities[ent_2_id]

    relation = format_relation(ent_1_text, ent_1_type, ent_2_text, ent_2_type, rel_type)
    offset = ent_1_end + ent_2_end
    to_pop = [ent_1_id, ent_2_id]

    return relation, offset, to_pop


def unpack_chunk(chunk: List[str]):
    lines = chunk.split("\n")

    title = lines[0].split("|")[-1]
    abstract = lines[1].split("|")[-1]

    entities = {}
    relations, offsets, ent_ids_to_pop = [], [], []

    for line in lines[2:]:
        if not line:
            continue
        annotation = line.split("\t")
        if len(annotation) > 4:  # this is an entity
            for entity in parse_entity(annotation):
                # Some entities don't have unique identifiers. Retain all of them.
                ent_start, ent_end, ent_text, ent_type, ent_id = entity
                if ent_id == "-1":
                    entities[ent_text] = (ent_start, ent_end, ent_text, ent_type)
                # Otherwise retain the first mention of each entity.
                elif ent_id not in entities:
                    entities[ent_id] = (ent_start, ent_end, ent_text, ent_type)
        else:  # this is a relation
            relation, offset, to_pop = parse_relation(annotation, entities)
            # Don't retain relations that are otherwise identical except for entity offsets or case.
            if relation.lower() not in map(str.lower, relations):
                relations.append(relation)
                offsets.append(offset)
                ent_ids_to_pop.extend(to_pop)

    # Remove entities already accounted for by relation annotations.
    for ent_id in ent_ids_to_pop:
        entities.pop(ent_id, None)

    entities = [format_entity(*entity[2:]) for entity in entities.values()]

    return title, abstract, entities, relations, offsets


def preprocess_bc5cdr(filepath: str) -> List[str]:

    # Step 1: Process the dataset chunk by chunk, collecting formatted relations and their offsets.
    processed_dataset = []
    chunks = Path(filepath).read_text().split("\n\n")
    with typer.progressbar(chunks, label="Processing") as progress:
        for chunk in progress:
            if not chunk:
                continue
            title, abstract, entities, relations, offsets = unpack_chunk(chunk)

            # Pack, sort, and unpack relations and their offsets
            relations = list(zip(*[relations, offsets]))
            relations.sort(key=itemgetter(1))
            relations, _ = list(zip(*relations))

            # Take the text to be the abstract and the title.
            text = f"{title} {abstract}"
            # Take the annotation to be the entities and the relations
            annotation = " ".join(entities + list(relations))

            processed_dataset.append(f"{text}\t{annotation}")

    return processed_dataset


@app.command(name="bc5cdr")
def main(input_dir: str, output_dir: str) -> None:
    train_filepath = Path(input_dir) / TRAIN_FILENAME
    dev_filepath = Path(input_dir) / VALID_FILENAME
    test_filepath = Path(input_dir) / TEST_FILENAME

    train = preprocess_bc5cdr(train_filepath)
    valid = preprocess_bc5cdr(dev_filepath)
    test = preprocess_bc5cdr(test_filepath)

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(train))
    (output_dir / "valid.tsv").write_text("\n".join(valid))
    (output_dir / "test.tsv").write_text("\n".join(test))


if __name__ == "__main__":
    app()
