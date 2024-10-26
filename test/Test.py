import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)  # Output one value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BatchModel(nn.Module):
  def __init__(self):
    super(BatchModel, self).__init__()
    self.fc1 = nn.Linear(512, 128)
    self.fc2 = nn.Linear(128, 1)  # Output one value per sample in the batch

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x



if __name__ == '__main__':
  print("--------- single row input example ---------------")
  # Example usage
  model = SimpleModel()
  input_sample = torch.randn(512)  # Single input of shape [512]
  print(f"{input_sample.shape = }, {input_sample.size = }, {input_sample = }")

  output = model(input_sample)
  print(output.shape)  # Output: torch.Size([1])
  print("--------- batch input example ---------------")
  # Example usage
  model = BatchModel()

  # Create a batch of 32 samples, each of shape [512]
  input_batch = torch.randn(32, 512)  # Batch of 32 inputs
  print(f"{input_batch.shape = }, {input_batch.size = }, {input_batch = }")
  # Forward pass with the entire batch
  output_batch = model(input_batch)
  print(output_batch.shape)  # Output: torch.Size([32, 1])

