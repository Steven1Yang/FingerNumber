import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 2x2池化层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 2x2池化层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2, 2),  # 2x2池化层
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = GestureRecognitionModel()
    print(model)
    input_tensor = torch.randn(2, 1, 224, 224)
    output = model(input_tensor)
    print(output.shape)
