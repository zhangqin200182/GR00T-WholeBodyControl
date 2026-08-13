"""Three diagnostic questions for step-0 2.04 rad error."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
from physx_env import PhysXEnv, BODY_NAMES

XML_PATH = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL_DIR  = "/sample_data/robot_filtered"

px.init_foundation()
env = PhysXEnv(px, XML_PATH, PKL_DIR, pos_iters=16, vel_iters=2, native_dt=0.001, decimation=20)

# Run 5 episodes for statistics
for ep in range(5):
    env.reset()
    ref_q0 = env.get_current_ref_qpos()
    actual_qpos = env.art.get_joint_positions()
    diff = actual_qpos - ref_q0
    absdiff = np.abs(diff)

    # Q1: Clamping at reset?
    if absdiff.max() > 0.01:
        print(f"\n=== Ep {ep}: CLAMP DETECTED ===")
        print(f"  max |actual - ref| = {absdiff.max():.4f} at joint {absdiff.argmax()}")
        # Find joints with clamp direction
        for j in range(29):
            if absdiff[j] > 0.01:
                sign = "↓clipped" if diff[j] < -0.01 else "↑clipped"
                print(f"  j{j}: ref={ref_q0[j]:.4f} actual={actual_qpos[j]:.4f} diff={diff[j]:.4f} {sign}")
    else:
        print(f"\n=== Ep {ep}: No clamping (max_diff={absdiff.max():.5f}) ===")

    # Q2: Per-joint errors at reset (after clamping fix or in initial state)
    action = (ref_q0 - env.jm) / env.jh
    action = np.clip(action, -1, 1)
    clipped_target = action * env.jh + env.jm
    target_diff = np.abs(ref_q0 - clipped_target)
    if target_diff.max() > 0.01:
        print(f"\n  Q2: Action clipping detected!")
        print(f"  max |ref - clipped_target| = {target_diff.max():.4f} at joint {target_diff.argmax()}")
        for j in range(29):
            if target_diff[j] > 0.01:
                print(f"  j{j}: ref={ref_q0[j]:.4f} target={clipped_target[j]:.4f} diff={ref_q0[j]-clipped_target[j]:.4f}")

    # Step 0: call step() and check what happens
    obs, reward, done, info = env.step(action.astype(np.float64))
    jp = env.art.get_joint_positions()
    step_diff = np.abs(jp - ref_q0)

    # Q3: termination check details
    rp = env.art.get_root_world_pose()[0]
    rq = env.art.get_root_world_pose()[1]
    rr = env._ref_root_pos()
    ref_quat = env._ref_root_quat()
    qe = np.linalg.norm(np.array(rq) - np.array(ref_quat))
    rh_diff = abs(rp[2] - rr[2])
    ori_err2 = qe**2
    h_thresh = 0.75 if rr[2] < 0.5 else 0.15

    # Ankle checks
    fk = env._fk
    tracked = fk.get_tracked_poses(rp, np.array([1,0,0,0], dtype=np.float64), jp)
    actual_body = np.array([t[0] for t in tracked])
    ref_body = env._ref_body_pos()

    la_idx = list(BODY_NAMES).index("left_ankle_roll_link")
    ra_idx = list(BODY_NAMES).index("right_ankle_roll_link")
    la_h_err = abs(ref_body[la_idx, 2] - actual_body[la_idx, 2])
    ra_h_err = abs(ref_body[ra_idx, 2] - actual_body[ra_idx, 2])
    ref_aligned_l = ref_body[la_idx] - rr + rp
    ref_aligned_r = ref_body[ra_idx] - rr + rp
    la_pos_err = np.linalg.norm(ref_aligned_l - actual_body[la_idx])
    ra_pos_err = np.linalg.norm(ref_aligned_r - actual_body[ra_idx])

    print(f"\n  Q3: Termination diagnostics (step 0 → done={done}):")
    print(f"  root_h_diff={rh_diff:.4f} (thresh={h_thresh}) {'← TERM' if rh_diff > h_thresh else ''}")
    print(f"  ori_err²={ori_err2:.4f} (thresh=0.5) {'← TERM' if ori_err2 > 0.5 else ''}")
    print(f"  LA_h_err={la_h_err:.3f} RA_h_err={ra_h_err:.3f} (thresh={h_thresh * 2.0}) ", end="")
    ank_h_fail = la_h_err > h_thresh * 2.0 or ra_h_err > h_thresh * 2.0
    print(f"{'← TERM' if ank_h_fail else 'OK'}")
    print(f"  LA_pos_err={la_pos_err:.3f} RA_pos_err={ra_pos_err:.3f} (thresh=0.5) ", end="")
    ank_p_fail = la_pos_err > 0.5 or ra_pos_err > 0.5
    print(f"{'← TERM' if ank_p_fail else 'OK'}")

    # Top joint errors at step 0
    print(f"  Top5 joint errors at step 0: ", end="")
    top5 = np.argsort(step_diff)[-5:]
    for j in top5[::-1]:
        print(f"j{j}:{step_diff[j]:.3f} ", end="")
    print()

    if done:
        print(f"  → Terminated at step 0!")

    if ep == 0:
        px.release_foundation()
        px.init_foundation()
        env = PhysXEnv(px, XML_PATH, PKL_DIR, pos_iters=16, vel_iters=2, native_dt=0.001, decimation=20)

px.release_foundation()
print("\nDone")
