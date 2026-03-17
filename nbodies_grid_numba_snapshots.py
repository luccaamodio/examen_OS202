"""Sequential (Numba-parallel) benchmark that saves position snapshots.

Usage example:

    NUMBA_NUM_THREADS=4 python3 nbodies_grid_numba_snapshots.py \
        data/galaxy_1000 0.001 20 20 1 50 10 snapshots/numba

Arguments:
    argv[1]: data file (default: data/galaxy_1000)
    argv[2]: dt (default: 0.001)
    argv[3], argv[4], argv[5]: n_cells in x,y,z (default: 20 20 1)
    argv[6]: n_steps (default: 100)
    argv[7]: snapshot_interval (default: 10)
    argv[8]: snapshot_dir (default: snapshots/numba)
"""

import os
import sys
import time
import numpy as np

from nbodies_grid_numba import NBodySystem


def main():
    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    n_steps = 100
    snapshot_interval = 10
    snapshot_dir = "snapshots/numba"

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    if len(sys.argv) > 6:
        n_steps = int(sys.argv[6])
    if len(sys.argv) > 7:
        snapshot_interval = int(sys.argv[7])
    if len(sys.argv) > 8:
        snapshot_dir = sys.argv[8]

    os.makedirs(snapshot_dir, exist_ok=True)

    system = NBodySystem(filename, ncells_per_dir=n_cells_per_dir)

    print(f"[numba] Simulation de {filename} avec dt = {dt}, grille {n_cells_per_dir}, "
          f"n_steps = {n_steps}, snapshots every {snapshot_interval} steps in {snapshot_dir}")

    times = []
    t0 = time.time()
    for step in range(n_steps):
        t1 = time.time()
        system.update_positions(dt)
        t2 = time.time()
        times.append(t2 - t1)

        if step % snapshot_interval == 0:
            snap_path = os.path.join(snapshot_dir, f"positions_step_{step:04d}.npy")
            np.save(snap_path, system.positions)

        if step % 10 == 0:
            print(f"[numba] Step {step}/{n_steps}", end="\r")

    t_total = time.time() - t0
    avg_update = sum(times) / len(times)
    print(f"\n[numba] Finished. Total time = {t_total:.3f} s, avg update = {avg_update*1000:.3f} ms")

    # Save timings metadata
    meta_path = os.path.join(snapshot_dir, "timings.txt")
    with open(meta_path, "w") as f:
        f.write(f"filename = {filename}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"n_cells_per_dir = {n_cells_per_dir}\n")
        f.write(f"n_steps = {n_steps}\n")
        f.write(f"snapshot_interval = {snapshot_interval}\n")
        f.write(f"total_time_s = {t_total}\n")
        f.write(f"avg_update_s = {avg_update}\n")


if __name__ == "__main__":
    main()
