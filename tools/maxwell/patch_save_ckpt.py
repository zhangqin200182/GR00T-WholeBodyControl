#!/usr/bin/env python3
"""Patch model_save_callback.py: tolerate deepcopy failures on NPU byte-storage tensors."""

path = "/data/z00666713/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/model_save_callback.py"
src = open(path).read()

if "NPU: deepcopy can fail" in src:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

old = """            _state = copy.deepcopy(state)
            _state.__dict__.pop("log_history")
"""
new = """            try:
                _state = copy.deepcopy(state)
            except Exception:
                # NPU: deepcopy can fail on byte-storage tensors (e.g. FSQ token
                # indices); fall back to shallow copy, torch.save stores current
                # values either way.
                _state = copy.copy(state)
            _state.__dict__.pop("log_history", None)
"""
count = src.count(old)
if count != 1:
    print(f"ANCHOR_COUNT_{count}")
    raise SystemExit(1)
open(path, "w").write(src.replace(old, new))
print("PATCHED")
