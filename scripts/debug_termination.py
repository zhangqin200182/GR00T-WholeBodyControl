"""Debug: trace termination reasons and PD tracking quality."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
from physx_env import PhysXEnv, BODY_NAMES
import physx_fk

XML_PATH = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL_DIR  = "/sample_data/robot_filtered"

px.init_foundation()
env = PhysXEnv(px, XML_PATH, PKL_DIR, pos_iters=16, vel_iters=2, native_dt=0.001, decimation=20)
env.reset()

# Get start info
start_qpos = env.get_current_ref_qpos()
print(f"Start: ref_qpos range [{start_qpos.min():.4f}, {start_qpos.max():.4f}]")

for step in range(20):
    ref_qpos = env.get_current_ref_qpos()

    # Check joint target displacement from initial
    if step == 0:
        prev_ref = start_qpos.copy()
    else:
        delta = ref_qpos - prev_ref
        max_delta = abs(delta).max()

    action = (ref_qpos - env.jm) / env.jh
    obs, reward, done, info = env.step(action.astype(np.float64))

    rp = env.art.get_root_world_pose()[0]
    rr = env._ref_root_pos()
    jp = env.art.get_joint_positions()
    jerr = np.linalg.norm(jp - ref_qpos)
    per_joint_err = np.abs(jp - ref_qpos)

    # Ankle tracking specifically
    fk = env._fk
    tracked = fk.get_tracked_poses(rp, np.array([1,0,0,0], dtype=np.float64), jp)
    actual_body = np.array([t[0] for t in tracked])
    ref_body = env._ref_body_pos()

    la_idx = list(BODY_NAMES).index("left_ankle_roll_link")
    ra_idx = list(BODY_NAMES).index("right_ankle_roll_link")
    la_err = np.linalg.norm(actual_body[la_idx] - ref_body[la_idx])
    ra_err = np.linalg.norm(actual_body[ra_idx] - ref_body[ra_idx])

    ref_aligned_l = ref_body[la_idx] - rr + rp
    ref_aligned_r = ref_body[ra_idx] - rr + rp
    aligned_err_l = np.linalg.norm(ref_aligned_l - actual_body[la_idx])
    aligned_err_r = np.linalg.norm(ref_aligned_r - actual_body[ra_idx])

    if step < 3 or done:
        print(f"  step {step}: z_err={abs(rp[2]-rr[2]):.4f}, jerr={jerr:.2f}, "
              f"max_jerr={per_joint_err.max():.2f}@{per_joint_err.argmax()}, "
              f"LA_err={la_err:.3f}, RA_err={ra_err:.3f}, "
              f"LA_aligned={aligned_err_l:.3f}, RA_aligned={aligned_err_r:.3f}, "
              f"done={done}")

    if done:
        print(f"\nTerminated at step {step}!")
        print(f"  root: actual=({rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f})")
        print(f"        ref   =({rr[0]:.3f},{rr[1]:.3f},{rr[2]:.3f})")
        print(f"  h_thresh={0.75 if rr[2]<0.5 else 0.15}")
        print(f"  joint range: [{jp.min():.4f}, {jp.max():.4f}]")
        print(f"  ref  range: [{ref_qpos.min():.4f}, {ref_qpos.max():.4f}]")
        print(f"  top 3 joint errors: ", end="")
        top3 = np.argsort(per_joint_err)[-3:]
        for j in top3[::-1]:
            print(f"j{j}:{per_joint_err[j]:.3f} ", end="")
        print()
        break

    prev_ref = ref_qpos.copy()

px.release_foundation()
