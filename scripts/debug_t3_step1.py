"""Detail: what happens in first physics step with T3 v2."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_fk, physx_loader
import joblib, glob, os

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk=list(pk.values())[0]
ref_qpos=pk["dof"][500].astype(np.float64)
ref_pos=np.array(pk["root_trans_offset"][500], dtype=np.float64)
# Lift slightly off ground to avoid initial ground penetration
ref_pos[2] += 0.1
pq=pk["root_rot"][500]; ref_quat=np.array([pq[3],pq[0],pq[1],pq[2]],dtype=np.float64)

px.init_foundation()

# Test with NO gravity first
print("=== Zero-gravity test ===")
art_zg=physx_loader.load_g1(px, XML)
sc_zg=px.create_scene(gravity=np.array([0,0,0],dtype=np.float32))
mat_zg=sc_zg.create_material(0.6,0.5,0.0)
sc_zg.add_ground_plane(mat_zg, np.array([0,0,1],dtype=np.float32))
sc_zg.add_articulation(art_zg)
art_zg.set_root_world_pose(ref_pos.astype(np.float32), ref_quat.astype(np.float32))
art_zg.set_joint_positions(ref_qpos.astype(np.float32))
art_zg.set_joint_velocities(np.zeros(29, dtype=np.float32))
art_zg.set_joint_drive_targets(ref_qpos.astype(np.float32))

# Check state before sim
rp=art_zg.get_root_world_pose()
jp=art_zg.get_joint_positions()
print(f"Before sim: root={rp[0]} ok={not np.any(np.isnan(rp[0]))} jp_ok={not np.any(np.isnan(jp))}")

# Run one step
sc_zg.simulate(0.002)
sc_zg.fetch_results()
rp=art_zg.get_root_world_pose()
jp=art_zg.get_joint_positions()
print(f"After sim: root={rp[0]} ok={not np.any(np.isnan(rp[0]))} jp_ok={not np.any(np.isnan(jp))}")
if np.any(np.isnan(jp)): print("  JOINT POSITIONS ARE NaN!")
if np.any(np.isnan(rp[0])): print("  ROOT POSITION IS NaN!")

# Check if root velocity exploded
rv=art_zg.get_root_world_velocity()
print(f"Root velocity: lin={rv[0]} ang={rv[1]}")
jv=art_zg.get_joint_velocities()
print(f"Joint vel range: [{jv.min():.4f}, {jv.max():.4f}]")

# Second step
sc_zg.simulate(0.002)
sc_zg.fetch_results()
rp=art_zg.get_root_world_pose()
jp=art_zg.get_joint_positions()
print(f"After sim 2: root={rp[0]} ok={not np.any(np.isnan(rp[0]))} jp_ok={not np.any(np.isnan(jp))}")

px.release_foundation()
