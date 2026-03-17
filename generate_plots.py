import os
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
SNAP = ROOT / "snapshots"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def parse_timings(path: Path):
    data = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [s.strip() for s in line.split("=", 1)]
            try:
                data[key] = float(val)
            except ValueError:
                data[key] = val
    return data


def plot_numba_threads():
    # For each dataset, collect avg_update_s vs threads
    datasets = [
        ("data/galaxy_1000", "1000"),
        ("data/galaxy_5000", "5000"),
        ("data/galaxy_10000", "10000"),
    ]
    threads = [1, 2, 4, 8, 12]

    plt.figure(figsize=(6, 4))

    for filename, label in datasets:
        y = []
        for t in threads:
            # base directory name pattern used previously
            if filename.endswith("1000"):
                base = f"numba_{t}threads"
            else:
                n = label
                base = f"numba_{t}threads_{n}"
            timing_path = SNAP / base / "timings.txt"
            if not timing_path.exists():
                y.append(None)
                continue
            info = parse_timings(timing_path)
            y.append(info.get("avg_update_s"))
        # filter out missing
        xs = [th for th, val in zip(threads, y) if val is not None]
        ys = [val for val in y if val is not None]
        if xs:
            plt.plot(xs, ys, marker="o", label=f"N = {label}")

    plt.xlabel("Threads numba")
    plt.ylabel("Temps moyen par update (s)")
    plt.title("Performance numba en fonction du nombre de threads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "numba_threads_vs_time.png"
    plt.savefig(out, dpi=150)
    plt.close()


def plot_mpi_comm_fraction():
    # Use timings from mpi_* directories for N=1000 and N=5000
    configs = [
        ("data/galaxy_1000", "mpi_4proc_t1", "N=1000, 4p, 1t"),
        ("data/galaxy_1000", "mpi_4proc_t2", "N=1000, 4p, 2t"),
        ("data/galaxy_1000", "mpi_4proc_t4", "N=1000, 4p, 4t"),
        ("data/galaxy_1000", "mpi_12proc_t1", "N=1000, 12p, 1t"),
        ("data/galaxy_5000", "mpi_4proc_t1_5000", "N=5000, 4p, 1t"),
        ("data/galaxy_5000", "mpi_4proc_t4_5000", "N=5000, 4p, 4t"),
        ("data/galaxy_5000", "mpi_12proc_t1_5000", "N=5000, 12p, 1t"),
    ]

    labels = []
    comm_fracs = []

    for filename, dirname, label in configs:
        tfile = SNAP / dirname / "timings.txt"
        if not tfile.exists():
            continue
        info = parse_timings(tfile)
        avg_step = info.get("avg_step_s")
        avg_comm = info.get("avg_comm_time_per_step_s_max")
        if avg_step and avg_comm is not None and avg_step > 0:
            frac = 100.0 * avg_comm / avg_step
            labels.append(label)
            comm_fracs.append(frac)

    if not labels:
        return

    plt.figure(figsize=(7, 4))
    x = range(len(labels))
    plt.bar(x, comm_fracs)
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel("Part du temps de communication (%)")
    plt.title("Fraction du temps de pas consacre\u0301e a\u0300 la communication MPI")
    plt.tight_layout()
    out = FIG_DIR / "mpi_comm_fraction.png"
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    plot_numba_threads()
    plot_mpi_comm_fraction()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
