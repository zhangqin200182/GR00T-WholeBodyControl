"""Task 1: 1-DOF revolute PD bandwidth benchmark.

Measures sinusoidal tracking bandwidth for Isaac leg PD (kp=99.1, kd=6.3)
on an isolated single joint.  Excludes multi-body coupling, joint friction,
and coordinate-frame errors.  Answers: "is kp=99.1 fast enough?"

After Task 1 bandwidth < 5Hz → Task 2 (PD sweep)
         bandwidth > 10Hz → Task 1.5 (joint frame audit)
"""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

AMP = 0.1         # rad — matches G1 reference motion step size
SIM_DT = 0.002    # 500 Hz
STEPS = 500        # per frequency
SETTLE = 300       # discard first N steps for steady-state analysis
FREQS = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]

def build_1dof(px, kp, kd, force_limit, inertia, mass=1.0):
    """Create 1-DOF revolute: fixed parent + child body with given inertia."""
    art = px.Articulation()
    art.add_link(-1, "parent")
    art.add_link(0, "child")
    art.add_joint(0, 1, 1, kp=kp, kd=kd, force_limit=force_limit)

    positions = np.array([0,0,0, 0,0,0], dtype=np.float32)  # zero offset
    quats     = np.array([1,0,0,0]*2, dtype=np.float32)
    parents   = np.array([-1, 0], dtype=np.int32)
    masses    = np.array([mass, mass], dtype=np.float32)
    inertias  = np.array([0.0]*3 + [inertia]*3, dtype=np.float32)  # only child has inertia
    com_pos   = np.zeros(6, dtype=np.float32)
    com_quat  = np.array([1,0,0,0]*2, dtype=np.float32)
    axis      = np.array([1], dtype=np.int32)   # X-axis rotation
    lower     = np.array([-3.14], dtype=np.float32)
    upper     = np.array([3.14], dtype=np.float32)
    fric      = np.array([0.0], dtype=np.float32)

    art.finalize(link_masses=masses, link_diag_inertia=inertias,
                 link_local_pos=positions, link_local_quat=quats,
                 link_parents=parents, link_com_pos=com_pos,
                 link_com_quat=com_quat,
                 joint_axis=axis, joint_lower=lower, joint_upper=upper,
                 joint_friction=fric, position_iters=8, velocity_iters=1)
    return art


def run_sweep(px, label, kp, kd, force_limit, inertia, mass=1.0):
    print(f"\n{'='*60}")
    print(f"  {label}: I={inertia:.4f}, kp={kp:.1f}, kd={kd:.2f}")
    print(f"{'='*60}")

    results = []
    for freq in FREQS:
        art = build_1dof(px, kp, kd, force_limit, inertia, mass)
        scene = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
        scene.add_articulation(art)

        # Fix parent at origin
        art.set_root_world_pose(np.array([0,0,0], dtype=np.float32),
                                np.array([1,0,0,0], dtype=np.float32))
        art.set_joint_positions(np.array([0.0], dtype=np.float32))
        art.set_joint_velocities(np.array([0.0], dtype=np.float32))

        errors = []
        for step in range(STEPS):
            t = step * SIM_DT
            target = AMP * np.sin(2 * np.pi * freq * t)
            art.set_joint_drive_targets(np.array([target], dtype=np.float32))
            scene.simulate(SIM_DT)
            scene.fetch_results()

            actual = art.get_joint_positions()[0]
            jp = np.array([actual])
            if np.any(np.isnan(jp)):
                print(f"    freq={freq:5.1f}Hz: NaN at step {step}")
                errors = []
                break
            e = actual - target
            if step >= SETTLE:
                errors.append(e)

        px.release_foundation()
        px.init_foundation()

        if errors:
            rms = np.sqrt(np.mean(np.array(errors)**2))
            gain = max(0.0, 1.0 - rms / AMP)  # tracking fidelity
            print(f"    freq={freq:5.1f}Hz:  RMS err={rms:.5f} rad  gain={gain:.3f}  ({'OK' if gain > 0.707 else 'LOW'})")
            results.append((freq, rms, gain))
        else:
            print(f"    freq={freq:5.1f}Hz:  FAILED (NaN)")
            results.append((freq, float('inf'), 0.0))

    # Find -3dB bandwidth
    bw = None
    for freq, rms, gain in results:
        if gain >= 0.707 and bw is None:
            pass  # not yet at -3dB
        elif gain < 0.707 and bw is None:
            bw = freq  # first freq below -3dB
    if bw is None and results:
        bw = results[-1][0]

    print(f"\n  >>> -3dB bandwidth ≈ {bw:.1f} Hz")

    if bw is not None:
        if bw < 5.0:
            print(f"  >>> VERDICT: PD bandwidth INSUFFICIENT → Task 2 (PD sweep)")
        else:
            print(f"  >>> VERDICT: PD bandwidth SUFFICIENT → Task 1.5 (joint frame audit)")

    return results, bw


px.init_foundation()

# ── Test 1: Leg-level inertia (~0.01, G1 thigh) ──
results_leg, bw_leg = run_sweep(
    px, "Leg inertia", kp=99.1, kd=6.3, force_limit=139.0,
    inertia=0.01, mass=2.0)

# ── Test 2: Arm-level inertia (~0.001, G1 forearm) ──
results_arm, bw_arm = run_sweep(
    px, "Arm inertia", kp=14.3, kd=0.9, force_limit=25.0,
    inertia=0.001, mass=0.5)

px.release_foundation()

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Leg (I=0.01, kp=99.1):  -3dB ≈ {bw_leg:.1f} Hz" if bw_leg else "  Leg: FAILED")
print(f"  Arm (I=0.001, kp=14.3): -3dB ≈ {bw_arm:.1f} Hz" if bw_arm else "  Arm: FAILED")

if bw_leg is not None and bw_leg < 5.0:
    print(f"\n  → NEXT: Task 2 (PD sweep)")
elif bw_leg is not None and bw_leg >= 10.0:
    print(f"\n  → NEXT: Task 1.5 (joint frame audit)")
else:
    print(f"\n  → AMBIGUOUS (5-10 Hz range): test with higher freq resolution")
print("Done")
