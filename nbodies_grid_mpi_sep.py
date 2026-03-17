"""MPI version: process 0 handles visualization, process 1 handles computation.

Usage (example):

    mpirun -np 2 python3 nbodies_grid_mpi_sep.py data/galaxy_5000 0.0015 15 15 1

Requires:
    - mpi4py installed
    - nbodies_grid_numba.py and visualizer3d.py in the same directory
"""

import sys
import numpy as np
from mpi4py import MPI

import visualizer3d
from nbodies_grid_numba import NBodySystem, generate_star_color


def read_initial_data(filename: str):
    """Read masses, positions, velocities and bounding box from a galaxy file."""
    positions = []
    velocities = []
    masses = []
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6],
                    [ 1.0e-6,  1.0e-6,  1.0e-6]], dtype=np.float64)

    with open(filename, "r") as f:
        for line in f:
            data = line.split()
            if not data:
                continue
            m = float(data[0])
            x, y, z = map(float, data[1:4])
            vx, vy, vz = map(float, data[4:7])
            masses.append(m)
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            for i in range(3):
                box[0, i] = min(box[0, i], positions[-1][i] - 1.0e-6)
                box[1, i] = max(box[1, i], positions[-1][i] + 1.0e-6)

    positions = np.array(positions, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    masses = np.array(masses, dtype=np.float32)
    return positions, velocities, masses, box


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("This MPI version requires at least 2 processes (0: visu, 1: compute).")
        return

    # Parse arguments on rank 0 then broadcast
    if rank == 0:
        filename = "data/galaxy_1000"
        dt = 0.001
        n_cells_per_dir = (20, 20, 1)
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        if len(sys.argv) > 2:
            dt = float(sys.argv[2])
        if len(sys.argv) > 5:
            n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        filename = None
        dt = None
        n_cells_per_dir = None

    filename = comm.bcast(filename, root=0)
    dt = comm.bcast(dt, root=0)
    n_cells_per_dir = comm.bcast(n_cells_per_dir, root=0)

    if rank == 0:
        # Visualization process
        positions, velocities, masses, box = read_initial_data(filename)
        max_mass = np.max(masses)
        colors = [generate_star_color(float(m)) for m in masses]
        intensity = np.clip(masses / max_mass, 0.5, 1.0)

        bounds = [[box[0, 0], box[1, 0]],
                  [box[0, 1], box[1, 1]],
                  [box[0, 2], box[1, 2]]]

        current_positions = positions.copy()

        def updater(local_dt: float):
            nonlocal current_positions
            # Send dt to compute process and receive new positions
            comm.send(local_dt, dest=1, tag=0)
            new_pos = comm.recv(source=1, tag=1)
            current_positions = new_pos.astype(np.float32, copy=False)
            return current_positions

        print(f"[rank 0] Simulation de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
        visu = visualizer3d.Visualizer3D(current_positions,
                                         colors,
                                         intensity,
                                         bounds)
        visu.run(updater=updater, dt=dt)

        # Tell compute process to stop
        comm.send(-1.0, dest=1, tag=0)

    elif rank == 1:
        # Compute process
        system = NBodySystem(filename, ncells_per_dir=n_cells_per_dir)
        print(f"[rank 1] Ready to compute orbits for {filename}")
        while True:
            local_dt = comm.recv(source=0, tag=0)
            if local_dt < 0.0:
                break
            system.update_positions(local_dt)
            comm.send(system.positions.copy(), dest=0, tag=1)

    else:
        # Extra ranks are idle but keep MPI alive
        while True:
            # Non-participating ranks just wait for a termination broadcast
            # Simple scheme: rank 0 never talks to them, so they immediately exit.
            break


if __name__ == "__main__":
    main()
