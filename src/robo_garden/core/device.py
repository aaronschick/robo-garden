"""GPU/CPU device detection and management."""

import logging

log = logging.getLogger(__name__)


def get_device_info() -> dict:
    """Detect available compute devices and return capabilities."""
    info = {"cpu": True, "cuda": False, "cuda_device_name": None, "vram_gb": None}

    # Check JAX CUDA availability (for MJX)
    try:
        import jax
        devices = jax.devices()
        for d in devices:
            if d.platform == "gpu":
                info["cuda"] = True
                info["cuda_device_name"] = str(d)
                break
        log.info(f"JAX devices: {devices}")
    except Exception as e:
        log.warning(f"JAX device detection failed: {e}")

    # Check PyTorch CUDA availability (for SB3/policy networks)
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_device_name"] = info["cuda_device_name"] or torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
            log.info(f"PyTorch CUDA: {info['cuda_device_name']}, {info['vram_gb']}GB VRAM")
    except Exception as e:
        log.warning(f"PyTorch device detection failed: {e}")

    return info


def recommend_num_envs(vram_gb: float | None = None) -> int:
    """Recommend number of parallel environments based on available VRAM."""
    if vram_gb is None:
        info = get_device_info()
        vram_gb = info.get("vram_gb")

    if vram_gb is None or vram_gb < 4:
        return 16  # CPU or very low VRAM
    elif vram_gb < 8:
        return 64
    elif vram_gb <= 12:
        return 128  # RTX 3070 (8GB) target
    elif vram_gb <= 24:
        return 512
    else:
        return 2048  # A100/H100 territory
