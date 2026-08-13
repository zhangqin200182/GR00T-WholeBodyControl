#!/usr/bin/env python3
"""ovphysx ref PD tracking test for G1 robot."""

import argparse, sys, os, time, glob
import numpy as np
import joblib

lib_paths = [
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib",
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin",
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins",
    "/usr/lib/aarch64-linux-gnu",
]
for p in lib_paths:
    e = os.environ.get("LD_LIBRARY_PATH", "")
    if p not in e:
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{e}"

import ovphysx, ovstage
from ovphysx.types import TensorType

FRAME_DT = 1.0 / 30.0
DECIMATION = 17
SIM_DT = FRAME_DT / DECIMATION

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


class Binding:
    def __init__(self, binding):
        self._b = binding
        self.shape = binding.shape
        self._buf = np.zeros(binding.shape, dtype=np.float32)

    def read(self):
        self._b.read(self._buf)
        return self._buf.copy()

    def write(self, data):
        self._buf[:] = data
        self._b.write(self._buf)

    def destroy(self):
        self._b.destroy()


def load_ref_data(pkl_dir, n_frames=50):
    pk = joblib.load(glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)[0])
    if "dof" not in pk:
        pk = list(pk.values())[0]
    ref = pk["dof"][500:500 + n_frames + 1].astype(np.float64)
    print(f"Ref data: {ref.shape[0]} frames, {ref.shape[1]} DOF")
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--pkl", default="/sample_data/robot_filtered")
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--decimation", type=int, default=DECIMATION)
    parser.add_argument("--print-per-joint", action="store_true")
    parser.add_argument("--pattern", default="/World/G1/*")
    args = parser.parse_args()

    physx, stage = None, None
    B = {}

    try:
        ovphysx.PhysX.set_cpu_mode(True)
        physx = ovphysx.PhysX()

        print(f"Loading USD: {args.usd}")
        stage = ovstage.Stage("g1-ref-pd")
        ovstage.population.open_usd(stage, str(args.usd), ordinal=1,
                                     domains=ovstage.PopulationDomain.PHYSICS)
        physx.attach_ovstage(stage, read_ordinal=1)
        physx.wait_all()

        for name, tt in [
            ("pos", TensorType.ARTICULATION_DOF_POSITION),
            ("vel", TensorType.ARTICULATION_DOF_VELOCITY),
            ("target", TensorType.ARTICULATION_DOF_POSITION_TARGET),
        ]:
            b = physx.create_tensor_binding(pattern=args.pattern, tensor_type=tt)
            B[name] = Binding(b)

        dof = B["pos"].shape[1]
        print(f"DOF: {dof}")

        ref_qpos = load_ref_data(args.pkl, n_frames=args.n_frames)

        # Init
        init = ref_qpos[0].astype(np.float32).reshape(1, -1)
        B["pos"].write(init)
        B["vel"].write(np.zeros((1, dof), dtype=np.float32))
        B["target"].write(init)

        for _ in range(20):
            physx.step(SIM_DT)
        physx.wait_all()

        # Track
        print(f"Tracking: {args.n_frames} frames, dec={args.decimation}, dt={SIM_DT:.6f}")
        t0 = time.time()
        errors = []

        for i in range(args.n_frames):
            target = ref_qpos[i + 1].astype(np.float32).reshape(1, -1)
            B["target"].write(target)

            for sub in range(args.decimation):
                physx.step(SIM_DT)
                if sub == args.decimation - 1:
                    actual = B["pos"].read().reshape(-1)
                    errors.append(actual - target.reshape(-1))

        physx.wait_all()
        elapsed = time.time() - t0

        errors = np.array(errors)
        alpha = float(np.sqrt(np.mean(errors ** 2)))

        print(f"\n=== RESULTS ===")
        print(f"Time: {elapsed:.1f}s ({elapsed/args.n_frames*1000:.1f} ms/frame)")
        print(f"alpha = {alpha:.6f} rad")

        if alpha < 0.002:
            print("EXCELLENT: matches Isaac target")
        elif alpha < 0.01:
            print(f"GOOD: {alpha/0.002:.1f}x Isaac target")
        elif alpha < 0.02:
            print(f"OK: near bare-SDK plateau (0.02)")
        else:
            print(f"BELOW bare-SDK best (0.02)")

        if args.print_per_joint:
            print("\nPer-joint RMS error:")
            for j in range(min(dof, len(JOINT_NAMES))):
                rms = np.sqrt(np.mean(errors[:, j] ** 2))
                bar = "#" * min(80, int(rms * 2000))
                print(f"  [{j:2d}] {JOINT_NAMES[j]:30s} {rms:.6f} {bar}")

    finally:
        for b in B.values():
            try: b.destroy()
            except: pass
        if stage is not None:
            try: physx.detach_ovstage(); stage.destroy()
            except: pass
        if physx is not None:
            try: physx.release()
            except: pass


if __name__ == "__main__":
    main()
