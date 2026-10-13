from datasets import load_from_disk
from pprint import pprint

path = "/root/autodl-tmp/ScholarAlign/data/raw/scitldr/hf_dataset"

dataset = load_from_disk(path)

print(dataset)

for split in dataset.keys():
    print("\n" + "=" * 80)
    print("Split:", split)
    print("Rows:", len(dataset[split]))
    print("Columns:", dataset[split].column_names)

    print("\nFirst sample:")
    pprint(dataset[split][0])