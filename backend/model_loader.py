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
from torchvision.models import resnet18, resnet34

# TreeSat species labels (20 classes from TreeSat benchmark)
SPECIES_LABELS = [
    "Abies", "Acer", "Alnus", "Betula", "Carpinus",
    "Castanea", "Corylus", "Fagus", "Fraxinus", "Larix",
    "Picea", "Pinus", "Populus", "Prunus", "Pseudotsuga",
    "Quercus", "Robinia", "Salix", "Sorbus", "Tilia"
]

NUM_SPECIES = len(SPECIES_LABELS)
INPUT_CHANNELS = 15  # S2 (13 bands) + S1 (2 bands)


class SqueezeExcitation(nn.Module):
    """Channel attention block for multispectral feature refinement."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale


class SpectralStem(nn.Module):
    """
    Spectral mixing stem for 15-channel Sentinel inputs.
    Improves channel fusion before feeding ResNet backbone.
    """

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            SqueezeExcitation(out_channels, reduction=4),
        )

    def forward(self, x):
        return self.stem(x)


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


class TreeSatMultiHeadModelV2(nn.Module):
    """
    V2 model:
    - Spectral mixing stem for multispectral channels
    - ResNet34 backbone
    - Stronger regularized heads with BN + SiLU + dropout
    """

    def __init__(self, in_channels=INPUT_CHANNELS, num_species=NUM_SPECIES):
        super().__init__()
        self.spectral_stem = SpectralStem(in_channels=in_channels, out_channels=32)

        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            32, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.density_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

        self.species_head = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_species),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.spectral_stem(x)
        features = self.backbone(x)
        density = self.density_head(features)
        species = self.species_head(features)
        return density, species


def build_model(variant: str, in_channels: int, num_species: int) -> nn.Module:
    """
    Build requested model variant.
    Supported values:
    - v1 (default): TreeSatMultiHeadModel
    - v2: TreeSatMultiHeadModelV2
    """
    key = (variant or "v1").strip().lower()
    if key in {"v2", "treesat_v2", "treesatmultiheadmodelv2"}:
        return TreeSatMultiHeadModelV2(in_channels=in_channels, num_species=num_species)
    return TreeSatMultiHeadModel(in_channels=in_channels, num_species=num_species)


def _load_state_dict_safe(model: nn.Module, state_dict: dict, strict_load: bool):
    """
    Attempt normal load first; on shape mismatch, fallback to loading only
    keys with matching names and tensor shapes.
    """
    try:
        model.load_state_dict(state_dict, strict=strict_load)
        return
    except Exception as e:
        print(f"[ModelLoader] Full state_dict load failed: {e}")
        model_state = model.state_dict()
        compatible = {}
        skipped = 0
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped += 1

        if not compatible:
            raise

        model.load_state_dict(compatible, strict=False)
        print(
            f"[ModelLoader] Loaded compatible subset: {len(compatible)} keys, skipped {skipped} keys"
        )


class ModelSingleton:
    """Singleton pattern for model loading - loads model once and reuses."""

    _instance = None
    _model = None
    _device = None
    _variant = None

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
        self._variant = os.getenv("MODEL_VARIANT", "v1")
        variant_key = (self._variant or "v1").strip().lower()
        self._model = build_model(
            variant=self._variant,
            in_channels=INPUT_CHANNELS,
            num_species=NUM_SPECIES,
        )

        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        explicit_checkpoint = os.getenv("MODEL_CHECKPOINT_PATH", "").strip()
        v2_load_checkpoint = os.getenv("MODEL_V2_LOAD_CHECKPOINT", "false").lower() == "true"
        is_v2 = variant_key in {"v2", "treesat_v2", "treesatmultiheadmodelv2"}
        should_load_checkpoint = (not is_v2) or v2_load_checkpoint or bool(explicit_checkpoint)

        # Try .pth file first, then directory format
        checkpoint_pth = os.path.join(checkpoints_dir, "best_model.pth")
        checkpoint_dir = os.path.join(checkpoints_dir, "best_model")

        checkpoint_path = None
        if explicit_checkpoint:
            checkpoint_path = explicit_checkpoint
        elif os.path.isfile(checkpoint_pth):
            checkpoint_path = checkpoint_pth
        elif os.path.isdir(checkpoint_dir):
            checkpoint_path = checkpoint_dir

        if should_load_checkpoint and checkpoint_path:
            print(f"[ModelLoader] Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            strict_load = os.getenv("MODEL_STRICT_LOAD", "false").lower() == "true"
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                _load_state_dict_safe(self._model, checkpoint["model_state_dict"], strict_load)
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                _load_state_dict_safe(self._model, checkpoint["state_dict"], strict_load)
            else:
                try:
                    _load_state_dict_safe(self._model, checkpoint, strict_load)
                except Exception as e:
                    print(f"[ModelLoader] Could not load state dict: {e}")
                    print("[ModelLoader] Using random init for demo")
        elif not should_load_checkpoint:
            print("[ModelLoader] Checkpoint loading disabled for current model variant/config")
            print("[ModelLoader] Set MODEL_V2_LOAD_CHECKPOINT=true or MODEL_CHECKPOINT_PATH to enable")
        else:
            print(f"[ModelLoader] No checkpoint found in {checkpoints_dir}")
            print("[ModelLoader] Using randomly initialized weights for demonstration")

        self._model.to(self._device)
        self._model.eval()
        print(f"[ModelLoader] Model variant: {self._variant}")
        print(f"[ModelLoader] Model loaded on {self._device}")

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None
        cls._model = None
        cls._device = None
        cls._variant = None
