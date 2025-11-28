import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        if x.size(0) != batch_size:
            print(f"警告: Batch Size 发生改变! 输入: {batch_size}, 展平后: {x.size(0)}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
if __name__ == "__main__":

    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"输入 Batch: 1, 输出 Batch: {output.shape[0]}")