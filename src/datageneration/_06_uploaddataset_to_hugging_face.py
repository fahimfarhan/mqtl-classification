from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
    sizes = [128, 256, 512, 515, 1024, 1027, 2048, 2051, 4096, 4099]
    splits = ["train", "validate", "test"]

    data_files = {}

    for size in sizes:
        folder_name = f"_{size}_"
        for split in splits:
            key = f"{split}_binned_{size}"
            filename = f"{folder_name}/_{size}_{split}_binned.csv"
            data_files[key] = filename

    print(f"{data_files = }")
    dataset = load_dataset("csv", data_files=data_files)

    # Push dataset to Hugging Face Hub
    dataset.push_to_hub("fahimfarhan/mqtl-classification-datasets")

