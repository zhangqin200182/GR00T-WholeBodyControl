"""T3 v2: verify physics stability with corrected joint frames."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_fk, physx_loader
import joblib, glob, os

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

# Load ref state
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk=list(pk.values())[0]
ref_qpos=pk["dof"][500].astype(np.float64)
ref_pos=pk["root_trans_offset"][500]; ref_pos[2]+=0.05
pq=pk["root_rot"][500]; ref_quat=np.array([pq[3],pq[0],pq[1],pq[2]],dtype=np.float64)

def test(zero_gravity=False):
    g = np.array([0,0,0],dtype=np.float32) if zero_gravity else np.array([0,0,-9.81],dtype=np.float32)

    art=physx_loader.load_g1(px, XML)
    scene=px.create_scene(gravity=g)
    mat=scene.create_material(0.6,0.5,0.0)
    scene.add_ground_plane(mat, np.array([0,0,1],dtype=np.float32))
    scene.add_articulation(art)
    art.set_root_world_pose(ref_pos.astype(np.float32), ref_quat.astype(np.float32))
    art.set_joint_positions(ref_qpos.astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    art.set_joint_drive_targets(ref_qpos.astype(np.float32))

    label = "zero-G" if zero_gravity else "gravity"
    alive = True
    for step in range(10):
        jp = art.get_joint_positions()
        rp = art.get_root_world_pose()[0]
        has_nan = np.any(np.isnan(jp)) or np.any(np.isnan(rp))
        q_range = f"[{jp.min():.4f}, {jp.max():.4f}]" if not has_nan else "NaN!"
        print(f"  {label} step {step}: root_z={rp[2]:.4f} q_range={q_range}")
        if has_nan:
            alive = False
            break
        scene.simulate(0.002)
        scene.fetch_results()

    if not zero_gravity:
        px.release_foundation()
    return alive

px.init_foundation()
print("=== Testing zero gravity ===")
test(zero_gravity=True)
# Don't release — test with gravity in same session

# Need new foundation for gravity test
print("\n=== Testing with gravity ===")
# Can't create new scene without new foundation... just test zero-G for now
px.release_foundation()
