"""Load NVIDIA SONIC release checkpoints saved by the TRL training fork.

The release checkpoint (e.g. /root/sonic_release/last.pt) contains objects
from a TRL fork (trl.trainer.utils.OnlineTrainerState) that do not exist in
the installed trl 0.28.  torch.load raises AttributeError on the missing
class; we inject a stub and retry, which is safe as long as the caller only
uses the state_dict keys.
"""
import re
import sys

import torch


def load_release(path, max_attempts=30):
    for attempt in range(max_attempts):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except AttributeError as e:
            msg = str(e)
            mm = re.search(r"attribute '(\w+)'", msg)
            mod = re.search(r"on <module '([^']+)'", msg)
            if mm and mod:
                name, modname = mm.group(1), mod.group(1)
                if modname in sys.modules and not hasattr(sys.modules[modname], name):
                    setattr(sys.modules[modname], name, type(name, (), {}))
                    continue
            raise
    raise RuntimeError(f"release checkpoint load failed: {path}")
