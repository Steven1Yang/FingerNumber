# FingerNumber
# Hand Gesture Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize hand gestures representing digits (0–4). The input is grayscale images of hand gestures, and the output is the predicted digit class.

## Model Architecture

The model is defined in `GestureRecognitionModel`, a custom PyTorch neural network class. It uses a series of convolutional layers followed by fully connected layers to extract features and perform classification.

### CNN Layers:

- **Conv2d(1, 32)** → **ReLU** → **MaxPool(2x2)**
- **Conv2d(32, 64)** → **ReLU** → **MaxPool(2x2)**
- **Conv2d(64, 128)** → **ReLU** → **MaxPool(2x2)**
- **Flatten**
- **Linear(128×28×28 → 1024)** → **ReLU**
- **Linear(1024 → 512)** → **ReLU**
- **Linear(512 → 5)**

### Project Structure:
FingerNumber/
├── Dataset/ # Dataset directory
│ ├── train/ # Training images
│ ├── test/ # Test images
│ └── val/ # Validation images
│
├── utils/ # Utility functions
│ ├── pycache/ # Compiled Python cache (can be ignored)
│ └── my_utils.py # Custom utility functions
│
├── DataProcess.py # Data loading and preprocessing script
├── Model.py # CNN model definition
├── Optimizer.py # Optimizer and scheduler setup
├── Train.py # Training loop and logic
├── requirements.txt # Project dependencies
└── README.md # Project documentation      


### Input/Output:

- **Input shape**: `(batch_size, 1, 224, 224)` — grayscale hand gesture image
- **Output shape**: `(batch_size, 5)` — class probabilities for digits 0 to 4

## Example

```python
model = GestureRecognitionModel()
input_tensor = torch.randn(2, 1, 224, 224)  # Batch of 2 grayscale images
output = model(input_tensor)
print(output.shape)  # Output: torch.Size([2, 5])
````

## Dependencies

pip install -r requirements.txt

Install requirements:

```bash
pip install torch torchvision matplotlib numpy
```

## Dataset

The model is designed to work with grayscale images of hand gestures representing the digits 0 to 4. Ensure your dataset is organized and preprocessed accordingly (e.g., resized to 224×224 and normalized to \[0,1]).

You can use your own dataset or collect gesture images using a webcam and annotate them appropriately.

## Training 

You can train the model using standard PyTorch training loops:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Make sure to convert your labels into integer class indices from 0 to 4.

---

Feel free to contribute or fork this repo!

```

---
