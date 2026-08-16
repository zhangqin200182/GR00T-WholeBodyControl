#!/usr/bin/env python3
"""1) Revert warp pre-import in eval_agent_trl.py. 2) Patch isaaclab fabric.py transform_compose for warp 1.7.1."""
import sys

# --- 1) revert eval_agent_trl.py warp pre-import ---
eval_path = "/root/GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py"
src = open(eval_path).read()
preimport = """# Pre-import pip warp-lang (newer than kit-bundled omni.warp 1.7.1, needed by isaaclab fabric.py)
import warp as _warp
import warp.types as _warp_types
if not hasattr(_warp_types, "array"):
    _warp_types.array = _warp.array

"""
if preimport in src:
    src = src.replace(preimport, "")
    open(eval_path, "w").write(src)
    print("EVAL_PREIMPORT_REVERTED")
elif "_warp_types" in src:
    print("EVAL_PREIMPORT_UNEXPECTED_STATE")
    sys.exit(1)
else:
    print("EVAL_ALREADY_CLEAN")

# --- 2) patch fabric.py ---
fabric_path = "/root/IsaacLab/source/isaaclab/isaaclab/utils/warp/fabric.py"
fsrc = open(fabric_path).read()

if "wp.quat_to_matrix" in fsrc:
    print("FABRIC_ALREADY_PATCHED")
    sys.exit(0)

old = """    # set transform matrix (need transpose for column-major ordering)
    # Using transform_compose as wp.transform_compose is deprecated
"""
# exact original text from the file (comment says wp.matrix() deprecated)
old = """    # set transform matrix (need transpose for column-major ordering)
    # Using transform_compose as wp.matrix() is deprecated
    fabric_matrices[fabric_index] = wp.mat44d(  # type: ignore[arg-type]
        wp.transpose(wp.transform_compose(position, rotation, scale))  # type: ignore[arg-type]
    )
"""
new = """    # set transform matrix (need transpose for column-major ordering)
    # NOTE: wp.transform_compose requires warp >= 1.16; kit bundles omni.warp 1.7.1,
    # so compose T*R*S manually (same layout as warp 1.16 transform_compose).
    rot = wp.quat_to_matrix(rotation)
    fabric_matrices[fabric_index] = wp.mat44d(  # type: ignore[arg-type]
        wp.transpose(
            wp.mat44(
                rot[0, 0] * scale[0], rot[0, 1] * scale[1], rot[0, 2] * scale[2], position[0],
                rot[1, 0] * scale[0], rot[1, 1] * scale[1], rot[1, 2] * scale[2], position[1],
                rot[2, 0] * scale[0], rot[2, 1] * scale[1], rot[2, 2] * scale[2], position[2],
                0.0, 0.0, 0.0, 1.0,
            )
        )
    )
"""
count = fsrc.count(old)
if count != 1:
    print(f"FABRIC_ANCHOR_COUNT_{count}")
    sys.exit(1)
open(fabric_path, "w").write(fsrc.replace(old, new))
print("FABRIC_PATCHED")
