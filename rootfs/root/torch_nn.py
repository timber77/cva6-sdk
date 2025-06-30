import os
os.environ["OPENBLAS_USE_HERO"] = "" # Unset here because i dont want any calls during import numpy to openblas/ hero inits
import torch
import torch.nn as nn

import perf

DTYPE = torch.float64

def hero_init():
    # Initialize hero
    print("Initializing hero")
    os.environ["OPENBLAS_USE_HERO"] = "1"
    perf.openblas_hero_init()
    print("Done")


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True, dtype=DTYPE)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        # print("Weights of fc1:", self.fc1.weight)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x
    
os.environ["HERO_USE_IOMMU"] = ""
hero_init()
test_data = torch.randn(10, 10, dtype=DTYPE)  # Example input tensor
model = SimpleNN(input_size=10, hidden_size=5, output_size=2)  # Example model
output = model(test_data)  # Forward pass
print("Output:", output)  # Print the output of the model
