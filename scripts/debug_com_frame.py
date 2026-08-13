import sys, numpy as np, xml.etree.ElementTree as ET
import joblib, glob, os
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_fk, physx_loader

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

tree = ET.parse(XML)
body_com = {}
def walk(elem):
    name = elem.get("name","")
    inertial = elem.find("inertial")
    pos=np.zeros(3); quat=np.array([1.,0,0,0])
    if inertial is not None:
        s=inertial.get("pos",""); sq=inertial.get("quat","")
        if s: parts=s.split(); pos=np.array([float(x) for x in parts[:3]])
        if sq: parts=sq.split(); quat=np.array([float(parts[0]),float(parts[1]),float(parts[2]),float(parts[3])])
    body_com[name] = {"pos":pos, "quat":quat}
    for c in elem:
        if c.tag=="body": walk(c)
walk(tree.find("worldbody").find("body"))

def _qi(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def _qm(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def _qr(q,v):
    qv=np.array([0,v[0],v[1],v[2]]); return _qm(_qm(q,qv),_qi(q))[1:]

# Load ref state
px.init_foundation()
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk=list(pk.values())[0]
ref_qpos=pk["dof"][500].astype(np.float64)
ref_pos=pk["root_trans_offset"][500]
pq=pk["root_rot"][500]; ref_quat=np.array([pq[3],pq[0],pq[1],pq[2]],dtype=np.float64)

# Raw articulation
art=physx_loader.load_g1(px, XML)
scene=px.create_scene(gravity=np.array([0,0,0],dtype=np.float32))
mat=scene.create_material(0.6,0.5,0.0)
scene.add_ground_plane(mat, np.array([0,0,1],dtype=np.float32))
scene.add_articulation(art)
art.set_root_world_pose(ref_pos.astype(np.float32), ref_quat.astype(np.float32))
art.set_joint_positions(ref_qpos.astype(np.float32))

# FK
fk=physx_fk.G1ForwardKinematics(XML)
all_poses=fk.compute(ref_pos, ref_quat, ref_qpos)

print("H1: PhysX=CoM subtract com→body origin vs FK")
print("H2: FK=body add com→CoM vs PhysX\n")

for i in range(art.num_links):
    name = fk.link_names[i]
    com=body_com.get(name, {"pos":np.zeros(3),"quat":np.array([1.,0,0,0])})
    px_pos=np.array(art.get_link_world_pose(i)[0])
    px_quat=np.array(art.get_link_world_pose(i)[1])
    fk_pos, fk_quat = all_poses[i]

    # H1: PhysX=CoM → body
    body_p = px_pos - _qr(px_quat, com["pos"])
    body_q = _qm(px_quat, _qi(com["quat"]))
    d1 = np.linalg.norm(body_p - fk_pos)

    # H2: FK=body → CoM
    com_p = fk_pos + _qr(fk_quat, com["pos"])
    d2 = np.linalg.norm(com_p - px_pos)

    d0 = np.linalg.norm(px_pos - fk_pos)

    h1="✓" if d1<0.05 else f"{d1:.3f}"
    h2="✓" if d2<0.05 else f"{d2:.3f}"
    if d0>0.01 or d1<0.05 or d2<0.05:
        print(f"  {name:35s}: raw={d0:7.3f}  H1={h1:>8}  H2={h2:>8}  com_pos={com['pos']}")

px.release_foundation()
