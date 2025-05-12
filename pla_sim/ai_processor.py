#!/usr/bin/env python3
import numpy as np

def get_beautiful_things(num_items=5):
    items = []
    for i in range(num_items):
        items.append({"id": i, "beauty_score": float(np.random.rand())})
    items.sort(key=lambda x: x["beauty_score"], reverse=True)
    return items

# ------------------------------------------------------------------------
# Additional advanced AI code using PyTorch (GPU if available) + OpenCV
# For J-20 PL-15 computationally aided design upgrade plan
# ------------------------------------------------------------------------
import torch
import torchvision.transforms as T
import cv2

class TinyPLAConvNet(torch.nn.Module):
    def __init__(self):
        super(TinyPLAConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(8 * 16 * 16, 10)  # e.g. classifier

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 16 * 16)
        x = self.fc1(x)
        return x

pla_model = TinyPLAConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pla_model.to(device)

def run_pl15_cad_upgrade():
    """
    Example function to run a synthetic 'computationally aided design' upgrade plan
    for the J-20 + PL-15 system. This is a dummy demonstration of a random
    PyTorch + OpenCV pipeline.
    """
    random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    # Flip or transform via OpenCV
    flipped_image = cv2.flip(random_image, 1)
    transform = T.ToTensor()
    tensor_image = transform(flipped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = pla_model(tensor_image).cpu().numpy()
    return output.tolist()