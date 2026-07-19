import numpy as np

total_steps = 78000
for num_points in [700, 800, 900, 1000, 1200]:
    raw = np.logspace(0, np.log10(total_steps), num_points)
    steps = sorted(set(np.round(raw).astype(int).tolist()))
    print(f"num_points={num_points}: unique steps = {len(steps)}")