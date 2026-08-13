"""Create v9 USD: add PhysxDrivePerformanceEnvelopeAPI to all joints."""
import re

path = '/root/GR00T-WholeBodyControl/g1_29dof_physx_v8.usda'
with open(path) as f:
    text = f.read()

# Replace apiSchemas on each joint to include PhysxDrivePerformanceEnvelopeAPI
# Current: prepend apiSchemas = ["PhysicsDriveAPI:angular"]
# Target:  prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxDrivePerformanceEnvelopeAPI"]
old = 'prepend apiSchemas = ["PhysicsDriveAPI:angular"]'
new = 'prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxDrivePerformanceEnvelopeAPI"]'
text = text.replace(old, new)

count = text.count('PhysxDrivePerformanceEnvelopeAPI')
print(f'Added PhysxDrivePerformanceEnvelopeAPI to {count} joints')

path_v9 = '/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda'
with open(path_v9, 'w') as f:
    f.write(text)
print(f'Written to {path_v9}')

# Verify a joint
elbow_block = re.search(r'def PhysicsRevoluteJoint "left_elbow_joint".*?\n\s+}', text, re.DOTALL)
if elbow_block:
    print(f'\nElbow joint:\n{elbow_block.group()}')
