from pathlib import Path
import torch

BIGNET_DIM = 1024

class LayerNorm(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[LayerNorm] Input Shape: {x.shape}")
        r = torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)
        print(f"[LayerNorm] Output Shape: {r.shape}\n")
        return r

class BigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
            )

        def forward(self, x):
            print(f"[Block] Input Shape: {x.shape}")
            out = self.model(x)
            print(f"[Block] Output Shape before Residual: {out.shape}")
            out += x  # Residual connection
            print(f"[Block] Output Shape after Residual: {out.shape}\n")
            return out

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

    def forward(self, x):
        print(f"[BigNet] Initial Input Shape: {x.shape}\n")
        return self.model(x)

def load(path: Path | None) -> BigNet:
    net = BigNet()
    if path is not None:
        print(f"[Loading Model] Loading weights from {path}")
        net.load_state_dict(torch.load(path, weights_only=True))
    return net

# Test Run
if __name__ == "__main__":
    net = BigNet()
    x = torch.randn(1, 1024)  # Example input (batch of 1)
    output = net(x)
    print(f"[Final Output] Shape: {output.shape}")
