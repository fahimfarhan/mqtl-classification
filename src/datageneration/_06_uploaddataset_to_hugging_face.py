from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
  data_files = {
    # small samples
    "train_binned_200": "dataset_200_train_binned.csv",
    "validate_binned_200": "dataset_200_validate_binned.csv",
    "test_binned_200": "dataset_200_test_binned.csv",
    # medium samples
    "train_binned_1000": "dataset_1000_train_binned.csv",
    "validate_binned_1000": "dataset_1000_validate_binned.csv",
    "test_binned_1000": "dataset_1000_test_binned.csv",
    # large samples
    "train_binned_4000": "dataset_4000_train_binned.csv",
    "validate_binned_4000": "dataset_4000_validate_binned.csv",
    "test_binned_4000": "dataset_4000_test_binned.csv",

  }
  dataset = load_dataset("csv", data_files=data_files)

  # Push dataset to Hugging Face hub
  dataset.push_to_hub("fahimfarhan/mqtl-classification-datasets")
  pass
