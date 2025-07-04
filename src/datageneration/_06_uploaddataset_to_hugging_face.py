from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
  data_files = {
    # small
    "train_binned_1027": "_1027_train_binned.csv",
    "validate_binned_1027": "_1027_validate_binned.csv",
    "test_binned_1027": "_1027_test_binned.csv",

    # medium
    "train_binned_2051": "_2051_train_binned.csv",
    "validate_binned_2051": "_2051_validate_binned.csv",
    "test_binned_2051": "_2051_test_binned.csv",

    # large
    "train_binned_4099": "_4099_train_binned.csv",
    "validate_binned_4099": "_4099_validate_binned.csv",
    "test_binned_4099": "_4099_test_binned.csv",

    # small
    "train_binned_1024": "_1024_train_binned.csv",
    "validate_binned_1024": "_1024_validate_binned.csv",
    "test_binned_1024": "_1024_test_binned.csv",

    # medium
    "train_binned_2048": "_2048_train_binned.csv",
    "validate_binned_2048": "_2048_validate_binned.csv",
    "test_binned_2048": "_2048_test_binned.csv",

    # large
    "train_binned_4096": "_4096_train_binned.csv",
    "validate_binned_4096": "_4096_validate_binned.csv",
    "test_binned_4096": "_4096_test_binned.csv",
  }
  dataset = load_dataset("csv", data_files=data_files)

  # Push dataset to Hugging Face hub
  dataset.push_to_hub("fahimfarhan/mqtl-classification-datasets")
  pass
