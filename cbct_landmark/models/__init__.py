"""Network definitions and checkpoint loading."""
import torch

from .unet_global import UNet3D_simple, UNet3D
from .unet_local import UNet3d

__all__ = ['UNet3D_simple', 'UNet3D', 'UNet3d', 'build_global_net', 'build_local_net', 'load_checkpoint']


def build_global_net():
    """Stage-1 tooth-region network exactly as used by the original ``tooth_region_pred.py``."""
    return UNet3D_simple(n_class=1)


def build_local_net():
    """Stage-2 landmark network exactly as used by the original ``2_train_and_valid.py``."""
    return UNet3d(n_class=1, act='relu')


def load_checkpoint(model, path, device='cpu', strict=True):
    """Load either checkpoint flavour produced by the original code.

    * ``tooth_best_model/bestmodel.pth``  -> plain ``state_dict``
    * ``UNet3d_stage1/Unet_model08.pt``    -> ``{'epoch', 'state_dict', 'optimizer_state_dict'}``
    ``DataParallel`` prefixes (``module.``) are stripped if present.
    """
    obj = torch.load(path, map_location=device)
    state = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    return model
