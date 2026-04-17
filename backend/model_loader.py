"""
TreeSat ResNet18 Multi-Head Model Loader
Handles model definition and singleton loading pattern.
- 15 input channels (Sentinel-2 + Sentinel-1 stacked)
- Density regression head (sigmoid output, 0-1)
- Multi-label species classification head (20 species)
"""

import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

# TreeSat species labels (20 classes from TreeSat benchmark)
SPECIES_LABELS = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Castanea", "Corylus", "Fagus", "Fraxinus", "Larix",
    "Picea", "Pinus", "Populus", "Prunus", "Pseudotsuga",
    "Quercus", "Robinia", "Salix", "Sorbus", "Tilia"
]

NUM_SPECIES = len(SPECIES_LABELS)
INPUT_CHANNELS = 15  # S2 (13 bands) + S1 (2 bands)


class TreeSatMultiHeadModel(nn.Module):
    """
    ResNet18-based multi-head model for TreeSat analysis.
    Head 1: Density regression (single scalar, sigmoid)
    Head 2: Multi-label species classification (20 species, sigmoid per class)
    """

    def __init__(self, in_channels=INPUT_CHANNELS, num_species=NUM_SPECIES):
        super().__init__()

        # Base ResNet18 backbone
        self.backbone = resnet18(weights=None)

        # Replace first conv layer: 3 channels -> 15 channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.fc.in_features

        # Remove the original FC layer
        self.backbone.fc = nn.Identity()

        # Head 1: Density regression
        self.density_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Head 2: Multi-label species classification
        self.species_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_species),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        density = self.density_head(features)
        species = self.species_head(features)
        return density, species


class ModelSingleton:
    """Singleton pattern for model loading - loads model once and reuses."""

    _instance = None
    _model = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_model(cls):
        instance = cls.get_instance()
        if instance._model is None:
            instance._load_model()
        return instance._model, instance._device

    def _load_model(self):
        """Load trained model from checkpoint."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TreeSatMultiHeadModel(
            in_channels=INPUT_CHANNELS,
            num_species=NUM_SPECIES
        )

        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

        # Try .pth file first, then directory format
        checkpoint_pth = os.path.join(checkpoints_dir, "best_model.pth")
        checkpoint_dir = os.path.join(checkpoints_dir, "best_model")

        checkpoint_path = None
        if os.path.isfile(checkpoint_pth):
            checkpoint_path = checkpoint_pth
        elif os.path.isdir(checkpoint_dir):
            checkpoint_path = checkpoint_dir

        if checkpoint_path:
            print(f"[ModelLoader] Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["state_dict"])
            else:
                try:
                    self._model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"[ModelLoader] Could not load state dict: {e}")
                    print("[ModelLoader] Using random init for demo")
        else:
            print(f"[ModelLoader] No checkpoint found in {checkpoints_dir}")
            print("[ModelLoader] Using randomly initialized weights for demonstration")

        self._model.to(self._device)
        self._model.eval()
        print(f"[ModelLoader] Model loaded on {self._device}")

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None
        cls._model = None
        cls._device = None
