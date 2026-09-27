"""
This is the CNN trained from scratch in Colab (see galaxeye_train_cnn.ipynb).
It goes directly from a 64x64 tile image to class logits - no separate
embedder + classifier head, unlike an earlier pretrained-backbone approach.

IMPORTANT: this class definition must stay IDENTICAL to the one used during
training (same layer names, sizes, order) - torch.load'ing a state_dict only
works if the architecture matches exactly. If you retrain with a different
architecture in Colab, update this class to match.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 64
ARTIFACTS_DIR = Path(__file__).parent / "artifacts_cnn"

# Same eval_transform used in the Colab notebook - NO augmentation at
# inference time, and no ImageNet normalization (this model was trained
# from scratch on raw [0,1] tensors).
_inference_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])


class SmallCNN(nn.Module):
    """Must exactly match the architecture trained in Colab."""

    def __init__(self, num_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class TileClassifier:
    """Loads the trained CNN once; classify() runs a single tile through it."""

    def __init__(self):
        with open(ARTIFACTS_DIR / "classes.json") as f:
            self.class_names = json.load(f)

        self.model = SmallCNN(num_classes=len(self.class_names))
        state_dict = torch.load(
            ARTIFACTS_DIR / "tile_cnn.pt", map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()  # important: disables dropout/batchnorm-update at inference

    @torch.no_grad()
    def classify(self, image: Image.Image) -> dict:
        x = _inference_transform(image.convert("RGB")).unsqueeze(0)  # [1,3,64,64]
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        prob_by_class = {
            cls: float(p) for cls, p in zip(self.class_names, probs)
        }
        predicted_class = max(prob_by_class, key=prob_by_class.get)
        return {
            "predicted_class": predicted_class,
            "confidence": prob_by_class[predicted_class],
            "all_class_probabilities": prob_by_class,
        }