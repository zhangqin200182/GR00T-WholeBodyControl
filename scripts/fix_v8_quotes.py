"""Fix v8 USD: add proper quotes to drive:type token values."""
import re

path = '/root/GR00T-WholeBodyControl/g1_29dof_physx_v8.usda'
with open(path) as f:
    text = f.read()

# Fix unquoted tokens: "= force" -> "= \"force\"" and "= acceleration" -> "= \"acceleration\""
text = text.replace(
    'uniform token drive:angular:physics:type = force',
    'uniform token drive:angular:physics:type = "force"'
)

# Write fixed version
path_v8b = '/root/GR00T-WholeBodyControl/g1_29dof_physx_v8.usda'
with open(path_v8b, 'w') as f:
    f.write(text)

# Verify
with open(path_v8b) as f:
    v8b = f.read()
count = v8b.count('uniform token drive:angular:physics:type = "force"')
print(f'Found {count} properly quoted drive:type attributes')

# Also verify a joint
import re as re2
elbow_block = re2.search(r'def PhysicsRevoluteJoint "left_elbow_joint".*?\n\s+}', v8b, re.DOTALL)
if elbow_block:
    print(f'\nElbow joint block:\n{elbow_block.group()}')
