"""Local runtime compatibility shims for the LCF3D workspace."""

import sys

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


# NumPy 2 serializes some objects from ``numpy._core``.  NumPy 1.x exposes
# the same implementation as ``numpy.core``; register aliases so checkpoints
# created in either environment remain readable without upgrading NumPy.
if np is not None and not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    for _module_name in ('multiarray', 'numeric', '_multiarray_umath'):
        _module = getattr(np.core, _module_name, None)
        if _module is not None:
            sys.modules.setdefault(f'numpy._core.{_module_name}', _module)

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    _original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        # MMEngine checkpoints in this workspace contain non-tensor metadata.
        # PyTorch 2.6+ defaults to weights_only=True, which breaks resume.
        if 'weights_only' not in kwargs or kwargs['weights_only'] is None:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
