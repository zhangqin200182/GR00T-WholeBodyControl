#!/usr/bin/env python3
"""Patch eval_agent_trl.py: pre-import pip warp-lang + shim old omni.warp APIs."""
import sys

path = "/root/GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py"
src = open(path).read()

old_block = """# Pre-import pip warp-lang (newer than kit-bundled omni.warp 1.7.1, needed by isaaclab fabric.py)
import warp as _warp
"""
new_block = """# Pre-import pip warp-lang (newer than kit-bundled omni.warp 1.7.1, needed by isaaclab fabric.py)
# + shim removed-in-1.16 APIs that isaacsim.core.utils.warp still references in annotations
import warp as _warp
import warp.types as _warp_types
if not hasattr(_warp_types, "array"):
    _warp_types.array = _warp.array
"""

if new_block.split("\n")[2] in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
if old_block in src:
    src = src.replace(old_block, new_block)
    open(path, "w").write(src)
    print("UPGRADED")
    sys.exit(0)

anchor = """try:
    import isaaclab  # noqa: F401
"""
count = src.count(anchor)
if count != 1:
    print(f"ANCHOR_COUNT_{count}")
    sys.exit(1)
open(path, "w").write(src.replace(anchor, new_block + "\n" + anchor))
print("PATCHED")
