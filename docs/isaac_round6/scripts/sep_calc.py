import numpy as np, glob
files = sorted(glob.glob('/tmp/isaac_r6/official_pinned_z*.npz')) + ['/tmp/isaac_r6/official_pinned_+0mm.npz']
for f in files:
    z = np.load(f)
    for side in ('left', 'right'):
        pz = z[f'pose_{side}_ankle_roll_link'][-10:, 2].mean()
        fz = np.linalg.norm(z[f'force_{side}_ankle_roll_link'][-10:], axis=1).mean()
        sep_lo = pz - 0.035  # sole with r=0.010 capsules
        sep_hi = pz - 0.033  # sole with r=0.008 capsules
        tag = f.split('/')[-1].replace('official_pinned_', '').replace('.npz', '')
        print(f"{tag:>12s} {side:5s} ankle_z={pz:.4f} sole_sep=[{sep_lo:+.4f},{sep_hi:+.4f}] F={fz:7.2f}N")
print('done')
