"""MPI + numba parallel version of the grid-based N-body simulation.

Each MPI rank holds a full copy of the system (positions, velocities, masses)
 but computes accelerations only for a disjoint subset of bodies. An MPI
 Allreduce is then used to assemble the full acceleration array on all ranks.
 NumPy/Numba are used inside each rank to exploit multi-threading.

Usage (example):

    mpirun -np 4 python3 nbodies_grid_mpi.py data/galaxy_5000 0.0015 20 20 1

This script focuses on parallelizing the *calculation* of trajectories.
Visualization can be done with a separate process (see nbodies_grid_mpi_sep.py)
 or by running a pure-visualization script that consumes snapshots.
"""

import sys
import os
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange

G = 1.560339e-13


@njit(parallel=False)
def update_stars_in_grid(cell_start_indices, body_indices,
                        cell_masses, cell_com_positions,
                        masses,
                        positions, grid_min, grid_max,
                        cell_size, n_cells):
    n_bodies = positions.shape[0]
    cell_start_indices.fill(-1)
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[morse_idx] += 1
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    for i in range(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position


@njit(parallel=True)
def compute_acceleration_slice(positions, masses,
                              cell_start_indices, body_indices,
                              cell_masses, cell_com_positions,
                              grid_min, grid_max,
                              cell_size, n_cells,
                              i_start, i_end):
    n_local = i_end - i_start
    a_local = np.zeros((n_local, 3), dtype=positions.dtype)
    for idx in prange(n_local):
        ibody = i_start + idx
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a_local[idx, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a_local[idx, :] += G * direction[:] * inv_dist3 * masses[jbody]
    return a_local


def read_initial_data(filename: str):
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


def decomposition(n_bodies: int, rank: int, size: int):
    base = n_bodies // size
    rem = n_bodies % size
    if rank < rem:
        i_start = rank * (base + 1)
        i_end = i_start + base + 1
    else:
        i_start = rem * (base + 1) + (rank - rem) * base
        i_end = i_start + base
    return i_start, i_end


def mpi_compute_acceleration(positions, masses,
                             cell_start_indices, body_indices,
                             cell_masses, cell_com_positions,
                             grid_min, grid_max,
                             cell_size, n_cells,
                             i_start, i_end,
                             comm):
    """Compute accelerations for a slice of bodies and assemble them globally.

    Returns both the full acceleration array and the time spent in
    computation vs communication (Allreduce), measured on this rank.
    """
    t0 = MPI.Wtime()

    # Update grid locally on each rank
    update_stars_in_grid(cell_start_indices, body_indices,
                         cell_masses, cell_com_positions,
                         masses,
                         positions, grid_min, grid_max,
                         cell_size, n_cells)

    a_local = compute_acceleration_slice(positions, masses,
                                         cell_start_indices, body_indices,
                                         cell_masses, cell_com_positions,
                                         grid_min, grid_max,
                                         cell_size, n_cells,
                                         i_start, i_end)

    t1 = MPI.Wtime()

    a_full = np.zeros_like(positions)
    a_full[i_start:i_end, :] = a_local

    t2 = MPI.Wtime()
    comm.Allreduce(MPI.IN_PLACE, a_full, op=MPI.SUM)
    t3 = MPI.Wtime()

    compute_time = t1 - t0
    comm_time = t3 - t2
    return a_full, compute_time, comm_time


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        filename = "data/galaxy_1000"
        dt = 0.001
        n_cells_per_dir = (20, 20, 1)
        n_steps = 100
        snapshot_interval = 10
        snapshot_dir = "snapshots/mpi"
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
        positions, velocities, masses, box = read_initial_data(filename)
    else:
        filename = None
        dt = None
        n_cells_per_dir = None
        n_steps = None
        snapshot_interval = None
        snapshot_dir = None
        positions = None
        velocities = None
        masses = None
        box = None

    filename = comm.bcast(filename, root=0)
    dt = comm.bcast(dt, root=0)
    n_cells_per_dir = comm.bcast(n_cells_per_dir, root=0)
    n_steps = comm.bcast(n_steps, root=0)
    snapshot_interval = comm.bcast(snapshot_interval, root=0)
    snapshot_dir = comm.bcast(snapshot_dir, root=0)

    positions = comm.bcast(positions, root=0)
    velocities = comm.bcast(velocities, root=0)
    masses = comm.bcast(masses, root=0)
    box = comm.bcast(box, root=0)

    n_cells = np.array(n_cells_per_dir, dtype=np.int64)
    grid_min = box[0].astype(np.float32)
    grid_max = box[1].astype(np.float32)
    cell_size = (grid_max - grid_min) / n_cells

    n_cells_tot = int(np.prod(n_cells))
    cell_start_indices = np.full(n_cells_tot + 1, -1, dtype=np.int64)
    body_indices = np.empty(positions.shape[0], dtype=np.int64)
    cell_masses = np.zeros(n_cells_tot, dtype=np.float32)
    cell_com_positions = np.zeros((n_cells_tot, 3), dtype=np.float32)

    n_bodies = positions.shape[0]
    i_start, i_end = decomposition(n_bodies, rank, size)
    if rank == 0:
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"MPI run with {size} processes, bodies per rank ~ {n_bodies / size:.1f}")
        print(f"Simulation de {filename} avec dt = {dt} et grille {n_cells_per_dir}, "
              f"n_steps = {n_steps}, snapshots every {snapshot_interval} steps in {snapshot_dir}")
        step_times = []

    # Accumulate local compute/communication times over all steps on each rank
    local_compute_time = 0.0
    local_comm_time = 0.0

    for step in range(n_steps):
        if rank == 0:
            t1 = time.time()

        a, comp1, comm1 = mpi_compute_acceleration(positions, masses,
                               cell_start_indices, body_indices,
                               cell_masses, cell_com_positions,
                               grid_min, grid_max,
                               cell_size, n_cells,
                               i_start, i_end,
                               comm)
        positions += velocities * dt + 0.5 * a * dt * dt
        a_new, comp2, comm2 = mpi_compute_acceleration(positions, masses,
                                   cell_start_indices, body_indices,
                                   cell_masses, cell_com_positions,
                                   grid_min, grid_max,
                                   cell_size, n_cells,
                                   i_start, i_end,
                                   comm)
        velocities += 0.5 * (a + a_new) * dt

        # Accumulate local times (two calls to mpi_compute_acceleration)
        local_compute_time += comp1 + comp2
        local_comm_time += comm1 + comm2

        if rank == 0:
            t2 = time.time()
            step_times.append(t2 - t1)
            if step % snapshot_interval == 0:
                snap_path = os.path.join(snapshot_dir, f"positions_step_{step:04d}.npy")
                np.save(snap_path, positions)
            if step % 10 == 0:
                print(f"Step {step}/{n_steps}", end="\r")

    # Réduire les temps de calcul et de communication à travers les rangs.
    # On prend le maximum, qui correspond au chemin critique (rank le plus lent).
    total_compute_time = comm.reduce(local_compute_time, op=MPI.MAX, root=0)
    total_comm_time = comm.reduce(local_comm_time, op=MPI.MAX, root=0)

    if rank == 0:
        print("\nSimulation MPI terminée.")
        total_time = sum(step_times)
        avg_step = total_time / len(step_times) if step_times else 0.0
        avg_compute = total_compute_time / n_steps if n_steps > 0 else 0.0
        avg_comm = total_comm_time / n_steps if n_steps > 0 else 0.0
        meta_path = os.path.join(snapshot_dir, "timings.txt")
        with open(meta_path, "w") as f:
            f.write(f"filename = {filename}\n")
            f.write(f"dt = {dt}\n")
            f.write(f"n_cells_per_dir = {n_cells_per_dir}\n")
            f.write(f"n_steps = {n_steps}\n")
            f.write(f"snapshot_interval = {snapshot_interval}\n")
            f.write(f"total_time_s = {total_time}\n")
            f.write(f"avg_step_s = {avg_step}\n")
            f.write(f"total_compute_time_s_max = {total_compute_time}\n")
            f.write(f"total_comm_time_s_max = {total_comm_time}\n")
            f.write(f"avg_compute_time_per_step_s_max = {avg_compute}\n")
            f.write(f"avg_comm_time_per_step_s_max = {avg_comm}\n")


if __name__ == "__main__":
    main()
