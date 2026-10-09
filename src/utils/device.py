"""Hardware acceleration device utility."""

def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
