from datasets import load_dataset

data_files = {
    "train": "https://huggingface.co/datasets/allenai/scitldr/resolve/refs%2Fconvert%2Fparquet/AIC/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/allenai/scitldr/resolve/refs%2Fconvert%2Fparquet/AIC/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/allenai/scitldr/resolve/refs%2Fconvert%2Fparquet/AIC/test/0000.parquet",
}

dataset = load_dataset(
    "parquet",
    data_files=data_files,
)

print(dataset)
print(dataset["train"][0])

dataset.save_to_disk(
    "/root/autodl-tmp/ScholarAlign/data/raw/scitldr/hf_dataset"
)

print("SciTLDR saved successfully.")