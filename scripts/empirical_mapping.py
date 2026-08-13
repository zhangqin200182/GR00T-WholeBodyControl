"""Empirically map tensor indices to DOF by working with raw tensors only."""
import sys, os, numpy as np
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

sys.path.insert(0, "/root/GR00T-WholeBodyControl")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")

import ovphysx, ovstage
from gear_sonic.envs.physx_env_ov import prepare_usd, TensorBinding, DOF2ACT, ACT2DOF
from ovphysx.types import TensorType

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("emp2")
usd = prepare_usd(ROBOT)

ovstage.population.open_usd(stage, usd, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"
b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

def step_cycle(n=17):
    for _ in range(n):
        px.step(0.001961)
    px.wait_all()

# Check DRIVE_MODEL
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
driven = []
for i in range(29):
    if np.any(dm[0, i, :] > 0):
        driven.append(i)
print(f"Driven tensor indices: {driven}")
print(f"Non-driven: {[i for i in range(29) if i not in driven]}")

# Map: for each tensor index, find which DOF order index it corresponds to
# We need to know the DOF order joint names to identify them
# But since we can't easily get joint names from the tensor, we'll build a pure
# tensor-index → tensor-index mapping (perturb target[t], see which position[s] moved)

print("\n=== Self-consistency: perturb target[t], check position[t] ===")
# First, write current positions as targets
pos0 = b_pos.read().ravel().copy()
b_tgt.write(pos0.astype(np.float32).reshape(1, -1))

# Wait for settle
for _ in range(5):
    step_cycle()

# Check: with matching pos=target, nothing should move
pos_ref = b_pos.read().ravel().copy()

# Now perturb each driven index one at a time
tensor_to_response = {}
for t_idx in range(29):
    # Re-sync: write positions as targets
    b_pos.write(pos0.astype(np.float32).reshape(1, -1))
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    b_tgt.write(pos0.astype(np.float32).reshape(1, -1))
    for _ in range(3):
        step_cycle()

    pos_before = b_pos.read().ravel().copy()
    tgt = pos_before.copy()
    tgt[t_idx] += 0.15
    b_tgt.write(tgt.astype(np.float32).reshape(1, -1))

    step_cycle()

    pos_after = b_pos.read().ravel()
    delta = np.abs(pos_after - pos_before)

    # Find which tensor indices moved (exclude the perturbed one itself since
    # it might move due to compliance/bounce)
    sorted_idx = np.argsort(-delta)
    top3 = [(j, float(delta[j])) for j in sorted_idx[:8] if delta[j] > 0.001]

    response_idx = sorted_idx[0] if delta[sorted_idx[0]] > 0.001 else -1
    self_ok = "SELF" if response_idx == t_idx else f"OTHER[{response_idx}]"
    no_response = "NO_RESP" if delta.max() < 0.001 else ""
    print(f"  tensor[{t_idx:2d}] → {self_ok} delta={delta.max():.4f} {no_response}  top: {top3}")

    if response_idx >= 0:
        tensor_to_response[t_idx] = response_idx

# Cleanup
for b in [b_pos, b_tgt, b_vel, b_dm]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()
print("\n*** DONE ***")
