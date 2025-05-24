from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
  data_files = {
    # small
    "train_binned_1029": "dataset_1029_train_binned.csv",
    "validate_binned_1029": "dataset_1029_validate_binned.csv",
    "test_binned_1029": "dataset_1029_train_binned.csv",

    # medium
    "train_binned_2053": "dataset_1029_train_binned.csv",
    "validate_binned_2053": "dataset_1029_validate_binned.csv",
    "test_binned_2053": "dataset_1029_test_binned.csv",

    # large
    "train_binned_4101": "dataset_1029_train_binned.csv",
    "validate_binned_4101": "dataset_1029_validate_binned.csv",
    "test_binned_4101": "dataset_1029_test_binned.csv",
  }
  dataset = load_dataset("csv", data_files=data_files)

  # Push dataset to Hugging Face hub
  dataset.push_to_hub("fahimfarhan/mqtl-classification-datasets")
  pass
