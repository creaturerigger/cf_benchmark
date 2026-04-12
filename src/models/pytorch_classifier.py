import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseModel

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    
}

OUTPUT_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=1),
}


class PYTModel(nn.Module, BaseModel):
    def __init__(self, in_features: int, cfg: dict):
        super(PYTModel, self).__init__()
        layers = []
        current_size = in_features
        for layer_cfg in cfg["model"]["layers"]:
            out = layer_cfg["out_features"]
            layers.append(nn.Linear(current_size, out))
            if layer_cfg.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(out))
            activation = layer_cfg.get("activation")
            if activation:
                layers.append(ACTIVATIONS[activation])
            dropout = layer_cfg.get("dropout", 0.0)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_size = out
        layers.append(nn.Linear(current_size, 1))
        output_activation = cfg["model"].get("output_activation")
        if output_activation:
            layers.append(OUTPUT_ACTIVATIONS[output_activation])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def load(cls, path: str | Path, cfg: dict) -> "PYTModel":
        """Load a previously saved model checkpoint.

        Args:
            path: path to the ``.pt`` file produced by ``Trainer.save_model``.
            cfg: the same config dict used at training time.

        Returns:
            A PYTModel with restored weights.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        # infer in_features from the first Linear layer
        first_weight = next(v for k, v in state_dict.items() if k.endswith(".weight"))
        in_features = first_weight.shape[1]
        model = cls(in_features=in_features, cfg=cfg)
        model.load_state_dict(state_dict)
        model.eval()
        return model
