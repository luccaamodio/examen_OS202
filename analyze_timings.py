import os
from pathlib import Path


def parse_timings(path: Path):
    data = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [s.strip() for s in line.split("=", 1)]
            # Try to parse numbers, keep string otherwise
            try:
                data[key] = float(val)
            except ValueError:
                data[key] = val
    return data


def summarize_snapshots(base: Path):
    rows = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        timing_file = sub / "timings.txt"
        if not timing_file.exists():
            continue
        t = parse_timings(timing_file)
        filename = t.get("filename", "?")
        avg_step = t.get("avg_step_s", None)
        avg_comp = t.get("avg_compute_time_per_step_s_max", None)
        avg_comm = t.get("avg_comm_time_per_step_s_max", None)
        comm_frac = None
        if avg_step and avg_comm is not None and avg_step > 0.0:
            comm_frac = avg_comm / avg_step
        rows.append({
            "dir": sub.name,
            "filename": filename,
            "avg_step_s": avg_step,
            "avg_compute_s": avg_comp,
            "avg_comm_s": avg_comm,
            "comm_fraction": comm_frac,
        })
    return rows


def main():
    base = Path("snapshots")
    if not base.exists():
        print("No snapshots directory found")
        return

    rows = summarize_snapshots(base)
    if not rows:
        print("No timings.txt files found under snapshots/")
        return

    # Print a simple table grouped by dataset
    by_file = {}
    for r in rows:
        by_file.setdefault(r["filename"], []).append(r)

    for filename, group in by_file.items():
        print(f"\n=== Dataset: {filename} ===")
        print("config\tavg_step_s\tavg_compute_s\tavg_comm_s\tcomm_frac")
        for r in sorted(group, key=lambda x: x["dir"]):
            cf = f"{r['comm_fraction']:.4f}" if r["comm_fraction"] is not None else "-"
            ac = f"{r['avg_compute_s']:.6f}" if r["avg_compute_s"] is not None else "-"
            am = f"{r['avg_comm_s']:.6f}" if r["avg_comm_s"] is not None else "-"
            astep = f"{r['avg_step_s']:.6f}" if r["avg_step_s"] is not None else "-"
            print(f"{r['dir']}\t{astep}\t{ac}\t{am}\t{cf}")


if __name__ == "__main__":
    main()
