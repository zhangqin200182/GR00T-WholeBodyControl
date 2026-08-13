"""Fix v9: add instance name to PhysxDrivePerformanceEnvelopeAPI."""
path = '/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda'
with open(path) as f:
    text = f.read()

old = 'PhysxDrivePerformanceEnvelopeAPI"]'
new = 'PhysxDrivePerformanceEnvelopeAPI:angular"]'
text = text.replace(old, new)
count = text.count('PhysxDrivePerformanceEnvelopeAPI:angular')
print(f'Fixed {count} joints')

with open(path, 'w') as f:
    f.write(text)

import re
joints = re.findall(r'prepend apiSchemas = \[(.*?)\]', text)
for j in joints:
    if 'Drive' in j and 'angular' in j:
        print(f'  apiSchemas = [{j}]')
        break
