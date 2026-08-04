#!/usr/bin/env python3
"""T2 API verification — PhysX 5 pybind11 wrapper unit tests."""
import sys, numpy as np

# Adjust path to find physx_core build
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px


def test_foundation():
    px.init_foundation()
    px.release_foundation()
    px.init_foundation()  # re-init
    print("  foundation: OK")


def test_scene():
    scene = px.create_scene(gravity=np.array([0, 0, -9.81], dtype=np.float32))
    assert isinstance(scene, px.Scene), f"expected Scene, got {type(scene)}"
    scene.simulate(0.002)
    scene.fetch_results()
    scene.fetch_results(False)
    print("  scene: OK")
    return scene


def test_quat_roundtrip():
    for q_in in [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0], dtype=np.float32),
        np.array([0, 0.707, 0, 0.707], dtype=np.float32),
    ]:
        q_out = px._test_quat_roundtrip(q_in)
        assert np.allclose(q_in, q_out, atol=1e-6), f"{q_in} -> {q_out}"
    print("  quat roundtrip: OK")


def test_articulation_simple():
    art = px.Articulation()
    art.add_link(-1, "base")
    art.add_link(0, "link1")
    art.add_joint(0, 1, 0, kp=100, kd=5, force_limit=50)

    n_links, n_joints = 2, 1
    art.finalize(
        link_masses=np.array([1.0, 0.5], dtype=np.float32),
        link_diag_inertia=np.array([0.01,0.01,0.01, 0.005,0.005,0.005], dtype=np.float32),
        link_local_pos=np.array([0,0,0, 0,0,-0.5], dtype=np.float32),
        link_local_quat=np.array([1,0,0,0, 1,0,0,0], dtype=np.float32),
        link_parents=np.array([-1, 0], dtype=np.int32),
        link_com_pos=np.zeros(6, dtype=np.float32),
        link_com_quat=np.array([1,0,0,0, 1,0,0,0], dtype=np.float32),
        joint_axis=np.array([0], dtype=np.int32),
        joint_lower=np.array([-1.57], dtype=np.float32),
        joint_upper=np.array([1.57], dtype=np.float32),
        joint_friction=np.array([0.1], dtype=np.float32),
    )
    assert art.num_links == n_links
    assert art.num_joints == n_joints
    print(f"  articulation simple: OK (links={art.num_links}, joints={art.num_joints})")
    return art


def test_joint_roundtrip(art):
    art.set_root_world_pose(
        np.array([0, 0, 1], dtype=np.float32),
        np.array([1, 0, 0, 0], dtype=np.float32))
    for val in [0.0, 0.5, -0.3, 1.57, -1.57]:
        art.set_joint_positions(np.array([val], dtype=np.float32))
        q = art.get_joint_positions()
        assert abs(q[0] - val) < 1e-5, f"set {val}, got {q[0]}"
    print("  joint roundtrip: OK")


def test_physics_step(scene, art):
    art.set_root_world_pose(
        np.array([0, 0, 1], dtype=np.float32),
        np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_joint_positions(np.array([0.5], dtype=np.float32))
    art.set_joint_velocities(np.array([0.0], dtype=np.float32))
    q0 = art.get_joint_positions()[0]
    for _ in range(100):
        scene.simulate(0.002)
        scene.fetch_results()
    q1 = art.get_joint_positions()[0]
    d = abs(q1 - q0)
    assert d > 1e-4, f"joint did not move under gravity (delta={d})"
    print(f"  physics: OK (joint moved {d:.4f} rad under gravity)")


def test_drive_targets(art):
    art.set_root_world_pose(
        np.array([0, 0, 1], dtype=np.float32),
        np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_joint_drive_targets(np.array([0.0], dtype=np.float32))
    art.set_joint_drive_params(0, kp=100, kd=5, force_limit=50)
    print("  drive targets: OK")


def test_link_pose(scene, art):
    art.set_root_world_pose(
        np.array([0, 0, 0.8], dtype=np.float32),
        np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_joint_positions(np.array([0.0], dtype=np.float32))
    scene.simulate(0.002); scene.fetch_results()
    root_pos, root_quat = art.get_root_world_pose()
    assert abs(root_pos[2] - 0.8) < 0.1, f"root z={root_pos[2]}"
    # Child link should be below root (z < root_z)
    child_pos, _ = art.get_link_world_pose(1)
    assert child_pos[2] < root_pos[2], f"child z={child_pos[2]} >= root z={root_pos[2]}"
    print(f"  link pose: OK (root.z={root_pos[2]:.2f}, child.z={child_pos[2]:.2f})")


def test_add_joint_validation():
    art = px.Articulation()
    art.add_link(-1, "base")
    art.add_link(0, "link1")
    # wrong child_idx should throw
    try:
        art.add_joint(0, 99, 0, kp=100, kd=5, force_limit=50)
        assert False, "should have thrown"
    except RuntimeError:
        pass
    print("  add_joint validation: OK")


def test_clean_shutdown(scene):
    px.release_foundation()
    print("  clean shutdown: OK")


if __name__ == "__main__":
    print("T2 API tests")
    px.init_foundation()
    test_quat_roundtrip()
    scene = test_scene()
    art = test_articulation_simple()
    test_joint_roundtrip(art)
    test_add_joint_validation()
    scene.add_articulation(art)
    test_physics_step(scene, art)
    test_drive_targets(art)
    test_link_pose(scene, art)
    test_clean_shutdown(scene)
    print("T2: ALL PASS")
