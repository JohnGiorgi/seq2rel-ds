import json
import random
from pathlib import Path

DRUG_LIST = "/Users/johngiorgi/Downloads/naacl2019/data/drug_lists/181109_ckb_all_drugs.txt"
EXAMPLES = "/Users/johngiorgi/Downloads/naacl2019/data/examples/document/ds_train_dev.txt"


def main():
    drug_list = Path(DRUG_LIST).read_text().strip().splitlines()
    examples = Path(EXAMPLES).read_text().strip().splitlines()

    # Build a list of known drug names
    drugs = set()
    for drug in drug_list:
        name, synonyms = drug.split("\t")
        drugs.add(name.lower().strip())
        drugs.update([syn.lower().strip() for syn in synonyms.split("|")])

    total_mentions = 0
    invalid_mentions = 0
    invalid_mentions_ex = []
    for example in examples:
        example = json.loads(example)
        for paragraph, mentions in zip(example["paragraphs"], example["mentions"]):
            for mention in mentions:
                # Only consider drug mentions
                if mention["type"] != "drug":
                    continue
                start, end = int(mention["start"]), int(mention["end"])
                mention_text = " ".join(paragraph[start:end]).lower().strip()
                if len(mention_text) <= 3 and mention_text not in drugs:
                    invalid_mentions += 1
                    invalid_mentions_ex.append(mention_text)
                total_mentions += 1
    print(f"{invalid_mentions / total_mentions:.2%} of drug mentions are invalid.")
    print(f"Examples of invalid mentions: {random.sample(invalid_mentions_ex, 10)}")


if __name__ == "__main__":
    main()
