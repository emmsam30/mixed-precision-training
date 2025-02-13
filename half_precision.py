from pathlib import Path
import torch
from .network import BIGNET_DIM, LayerNorm  # noqa: F401

class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Half-precision Linear Layer that performs computations in float16
        while maintaining float32 input/output compatibility.
        """
        super().__init__(in_features, out_features, bias)
        # Convert weights to half precision
        self.weight.data = self.weight.data.to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.float16)
        # Disable gradient computation
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype (typically float32)
        initial_data_type = x.dtype
        # Convert to half precision for computation
        x_float_16 = x.to(torch.float16)
        # Compute in half precision
        output = torch.nn.functional.linear(x_float_16, self.weight, self.bias)
        # Convert back to original dtype
        output = output.to(initial_data_type)
        return output

class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Normalization layers
    remain in full precision to avoid numerical instability.
    """
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path | None) -> HalfBigNet:
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net