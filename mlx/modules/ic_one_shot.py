import torch
from mlx.modules.ic.siamese_le_net import SiameseLeNet

def run_ic_one_shot(model, **kwargs):
    defaults = {
        "action": "test",
        "device": "cpu",
        "embedding_size": 4096,
        "input_size": (105, 105),
        "batch_size": 1
    }

     # Merge defaults with kwargs (user overrides)
    config = {**defaults, **kwargs}

    if model == "siamese-le-net":
        net = SiameseLeNet(colored=True, embedding_size=config["embedding_size"])
    else:
        raise ValueError(f"Invalid model {model}")

    if config["action"] == "test":
        _test_model(net, config)
    else:
        raise ValueError(f"Unsupported action {config['action']}")

def _test_model(net, config):
    """Run test with random tensors using resolved config."""
    batch   = config["batch_size"]
    h, w    = config["input_size"]
    device  = config["device"]

    print(f"Running test on device={device} | input={h}x{w} | batch={batch}")

    # Assume colored
    x1 = torch.randn(batch, 3, h, w).to(device)
    x2 = torch.randn(batch, 3, h, w).to(device)

    out = net(x1, x2)
    print(f"Output shape: {out.shape}")
    print(out)
