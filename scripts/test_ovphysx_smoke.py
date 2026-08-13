#!/usr/bin/env python3
"""Smoke test: load G1 USD into ovphysx and verify articulation DOF access."""

import argparse, sys, os, time
import numpy as np

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


class Binding:
    """Thin wrapper around ovphysx TensorBinding with pre-allocated buffers."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--pattern", default="/World/G1/pelvis")
    args = parser.parse_args()

    print("=" * 60)
    print("ovphysx G1 Smoke Test")
    print("=" * 60)

    physx, stage = None, None
    B = {}

    try:
        print("\n[1] Creating PhysX...")
        ovphysx.PhysX.set_cpu_mode(True)
        physx = ovphysx.PhysX()
        print("    OK")

        print(f"\n[2] Loading USD: {args.usd}")
        stage = ovstage.Stage("g1-smoke")
        ovstage.population.open_usd(stage, str(args.usd), ordinal=1,
                                     domains=ovstage.PopulationDomain.PHYSICS)
        physx.attach_ovstage(stage, read_ordinal=1)
        physx.wait_all()
        print("    OK")

        print(f"\n[3] Tensor bindings (pattern: {args.pattern})...")
        for name, tt in [
            ("POS", TensorType.ARTICULATION_DOF_POSITION),
            ("VEL", TensorType.ARTICULATION_DOF_VELOCITY),
            ("TARGET", TensorType.ARTICULATION_DOF_POSITION_TARGET),
        ]:
            b = physx.create_tensor_binding(pattern=args.pattern, tensor_type=tt)
            B[name] = Binding(b)
            print(f"    {name}: shape={b.shape}")

        dof = B["POS"].shape[1]
        print(f"\n    DOF count: {dof}")
        print(f"    Initial: {B['POS'].read().ravel()[:8]}")

        # Set knee targets and step
        print("\n[4] Step test: target knees=0.5 rad, 30 steps @ dt=0.002")
        target = np.zeros((1, dof), dtype=np.float32)
        target[0, 3] = 0.5
        target[0, 9] = 0.5
        B["TARGET"].write(target)

        for _ in range(30):
            physx.step(0.002)
        physx.wait_all()

        pos = B["POS"].read()
        print(f"    knee_l={pos[0,3]:.6f}  knee_r={pos[0,9]:.6f}  (target=0.5)")

        # Bench
        print("\n[5] Timing 100 steps...")
        t0 = time.time()
        for _ in range(100):
            physx.step(0.002)
        physx.wait_all()
        ms = (time.time() - t0) / 100 * 1000
        print(f"    {ms:.2f} ms/step")

        print("\n" + "=" * 60)
        print(f"SMOKE TEST PASSED  |  DOF={dof}  step={ms:.2f}ms")
        print("=" * 60)

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
