from .mlp import MLP
from .cnn import CNN
from .resnet import MiniResNet


def build_model(config: dict):
    """Factory: pick model from config['model'] string."""
    name = config["model"].lower()
    if name == "mlp":
        return MLP(
            input_dim=config.get("input_dim", 784),
            hidden_dims=config.get("hidden_dims", [256, 128]),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.2),
        )
    if name == "cnn":
        return CNN(
            in_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 10),
        )
    if name in ("resnet", "miniresnet"):
        return MiniResNet(
            num_classes=config.get("num_classes", 200),
            in_channels=config.get("in_channels", 3),
        )
    raise ValueError(f"Unknown model: {name!r}. Choose from: mlp, cnn, resnet")
