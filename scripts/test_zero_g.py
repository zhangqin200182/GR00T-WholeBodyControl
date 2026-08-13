"""Test PD tracking in zero gravity — compare BFS vs identity mapping."""
import sys, os, numpy as np, tempfile
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

sys.path.insert(0, "/root/GR00T-WholeBodyControl")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")

from gear_sonic.envs.physx_env_ov import TensorBinding
from ovphysx.types import TensorType
import ovphysx, ovstage

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"

# Build zero-G USD
with open(ROBOT) as rf:
    original = rf.read()
world_open = 'def Xform "World"'
insert_pos = original.index(world_open) + len(world_open)
brace_pos = original.index("{", insert_pos)
zero_g_scene = '\n    def PhysicsScene "physicsScene"\n    {\n        float3 gravity = (0, 0, 0)\n    }\n'
combined = original[:brace_pos + 1] + zero_g_scene + original[brace_pos + 1:]

with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False) as f:
    f.write(combined)
    zero_g_path = f.name

# Two mappings to test
MAPPINGS = {
    "BFS":      np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28], dtype=np.int32),
    "IDENTITY": np.arange(29, dtype=np.int32),
}

native_dt = 0.001961
decimation = 17
ctrl_dt = native_dt * decimation

for name, dof2act in MAPPINGS.items():
    act2dof = np.zeros(29, dtype=np.int32)
    for d, a in enumerate(dof2act):
        act2dof[a] = d

    print(f"\n{'='*60}")
    print(f"=== {name} mapping ===")
    print(f"DOF2ACT[:10] = {dof2act[:10].tolist()}...")

    ovphysx.PhysX.set_cpu_mode(True)
    px = ovphysx.PhysX()
    stage = ovstage.Stage(f"zg_{name}")

    ovstage.population.open_usd(stage, zero_g_path, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
    px.attach_ovstage(stage, read_ordinal=1)
    px.wait_all()

    pattern = "/World/G1/*"
    b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
    b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
    b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

    def read_dof():
        return b_pos.read().ravel()[act2dof].astype(np.float64)
    def write_dof(q):
        b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))
    def write_tgt(t):
        b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))

    def step(n=decimation):
        for _ in range(n):
            px.step(native_dt)
        px.wait_all()

    # Test 1: Hold at zero in zero-G
    print("  Test 1: Zero hold...")
    zero = np.zeros(29, dtype=np.float64)
    write_dof(zero)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    write_tgt(zero)
    px.update_articulations_kinematic()
    step(5)
    q = read_dof()
    delta = np.abs(q)
    print(f"    max={delta.max():.6f} rad  mean={delta.mean():.6f} rad")

    # Test 2: per-DOF perturbation at zero pose (small angles = safe)
    print("  Test 2: Per-DOF perturbation...")
    match_count = 0
    for dof_i in range(29):
        write_dof(zero)
        write_tgt(zero)
        px.update_articulations_kinematic()
        step(1)
        pos_before = read_dof()
        tgt = np.zeros(29, dtype=np.float64)
        tgt[dof_i] = 0.5  # 0.5 rad perturbation
        write_tgt(tgt)
        step(1)
        q = read_dof()
        delta = np.abs(q - pos_before)
        worst = np.argmax(delta)
        if worst == dof_i: match_count += 1
    print(f"    {match_count}/29 MATCH")

    # Test 3: Sinusoidal tracking (small amplitude, slow frequency)
    print("  Test 3: Sine tracking (A=0.1 rad, f=1 Hz)...")
    # Reset to zero
    write_dof(zero)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    write_tgt(zero)
    px.update_articulations_kinematic()
    step(3)

    n_frames = 100
    errors = []
    t = 0.0
    for i in range(n_frames):
        t += ctrl_dt
        # Only drive a few joints with sine to reduce coupling
        ref_target = np.zeros(29, dtype=np.float64)
        ref_target[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)  # hip_pitch
        ref_target[6] = 0.1 * np.sin(2 * np.pi * 1.0 * t + 0.5)  # right_hip
        ref_target[18] = 0.1 * np.sin(2 * np.pi * 1.5 * t)  # left_elbow
        write_tgt(ref_target)
        step()
        q_actual = read_dof()
        errors.append(q_actual - ref_target)

    errors = np.array(errors)
    # Only evaluate on driven joints
    driven_joints = [0, 6, 18]
    mask = np.zeros(29, dtype=bool)
    mask[driven_joints] = True
    alpha_driven = float(np.sqrt(np.mean(errors[-50:, mask] ** 2)))
    print(f"    alpha (driven joints only) = {alpha_driven:.6f} rad")

    # Per-joint RMS on last 50 frames
    err_tail = errors[-50:]
    for j in driven_joints:
        rms = np.sqrt(np.mean(err_tail[:, j] ** 2))
        print(f"      DOF[{j:2d}] rms={rms:.6f}")

    # Cleanup
    for b in [b_pos, b_tgt, b_vel]:
        try: b.destroy()
        except: pass
    px.detach_ovstage()
    px.release()

os.unlink(zero_g_path)
print("\n*** DONE ***")
