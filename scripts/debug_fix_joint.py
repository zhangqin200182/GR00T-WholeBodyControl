"""Fix: set correct joint parentPose from Python side, then test FK + physics."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)  # Y axis

positions = np.array([0,0,1.0, 0,0,-0.5], dtype=np.float32)
quats     = np.array([1,0,0,0]*2, dtype=np.float32)
parents   = np.array([-1, 0], dtype=np.int32)
masses    = np.array([1.0, 0.5], dtype=np.float32)
inertias  = np.array([0.01]*6, dtype=np.float32)
com_pos   = np.zeros(6, dtype=np.float32)
com_quat  = np.array([1,0,0,0]*2, dtype=np.float32)
axis      = np.array([1], dtype=np.int32)
lower     = np.array([-3.14], dtype=np.float32)
upper     = np.array([3.14], dtype=np.float32)
fric      = np.array([0.0], dtype=np.float32)

art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

# Now fix joint frame from Python — can we access the joint?
# The PhysX joint is not directly exposed to Python...
# Let me check if we need C++ changes.

# Test: what if we pass LOCAL pos (not world) to createLink?
# But local pos = child_local in parent frame = [0,0,-0.5]
# If we compute child_world = parent_world * local AND parentPose = parent⁻¹ * child_world,
# then parentPose = parent⁻¹ * parent * local = local. That's correct!
# But parent⁻¹ * child_world must compute correctly...

# The issue is that createLink computes parentPose in CoM frame without accounting for
# the parent's world pose. "transformInv(pose)" with CoM=identity just gives the world pose back.

# Let's verify: what parentPose does createLink compute internally?
# If parentPose = Identity.transformInv(child_world) = Identity⁻¹ * (0,0,0.5) = (0,0,0.5)
# Then child_position_in_chain = parent + parentPose = (0,0,1)+(0,0,0.5) = (0,0,1.5)
# This matches observation A!

# To fix: we need parentPose = parent⁻¹ * child_world = (0,0,-0.5)
# Solution: we need to compute the correct local transform and pass it to setParentPose.

# Since setParentPose is on PxArticulationJointReducedCoordinate which we don't expose to Python,
# we need to fix this in C++. But the earlier C++ fix caused simulation crashes.

# ALTERNATIVE: Don't fix parentPose. Instead, when computing FK positions,
# we need to compensate for the wrong parentPose.
# Each link's position = parent + rotation(parentPose_wrong)
# parentPose_wrong = child_world_when_created
# We can't use parent_CoM to convert because CoM is zeroed.

# Actually the correct approach: PASS THE LOCAL POSITION to createLink!
# NOT the world position. But that caused NaN in the original test!

# Wait, maybe the NaN was caused by the old code (before world-space accumulate was added).
# Let me try: pass LOCAL positions to createLink

print("=== Test LOCAL positions ===")
art2 = px.Articulation()
art2.add_link(-1, "parent2")
art2.add_link(0, "child2")
art2.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

# Pass LOCAL positions (not world-accumulated)
positions_local = np.array([0,0,1.0, 0,0,-0.5], dtype=np.float32)
art2.finalize(link_masses=masses, link_diag_inertia=inertias,
              link_local_pos=positions_local, link_local_quat=quats,
              link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
              joint_axis=axis, joint_lower=lower, joint_upper=upper,
              joint_friction=fric, position_iters=8, velocity_iters=1)

# Add to scene and check
scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
scene2.add_ground_plane(scene2.create_material(0.6,0.5,0.0), np.array([0,0,1], dtype=np.float32))
scene2.add_articulation(art2)

print(f"  parent: {art2.get_link_world_pose(0)[0]}")
print(f"  child:  {art2.get_link_world_pose(1)[0]}")
# If local poses work: parent at (0,0,1), child at (0,0,1)+(0,0,-0.5)=(0,0,0.5)

art2.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(np.array([0.0], dtype=np.float32))
print(f"  After set_root(z=3): parent={art2.get_link_world_pose(0)[0]} child={art2.get_link_world_pose(1)[0]}")
# Expected: child at (0,0,2.5)

px.release_foundation()
print("Done")
