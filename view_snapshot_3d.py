"""Visualiser un snapshot .npy de positions avec le visualiseur 3D.

Usage :

    /Users/lucca-amodio/Documents/Sistemas_paralelos/prova/.venv/bin/python \
        view_snapshot_3d.py snapshots/numba_4threads/positions_step_0030.npy \
        data/galaxy_1000

Arguments :
    1) chemin vers le fichier .npy de positions (N,3)
    2) fichier de données d'origine (pour récupérer les masses et définir les couleurs)
"""

import sys
import numpy as np

import visualizer3d


def generate_star_color(mass: float) -> tuple[int, int, int]:
    """Même logique de couleur que dans nbodies_grid_numba.py."""
    if mass > 5.0:
        return (150, 180, 255)  # bleu-blanc
    elif mass > 2.0:
        return (255, 255, 255)  # blanc
    elif mass >= 1.0:
        return (255, 255, 200)  # jaune
    else:
        return (255, 150, 100)  # rouge-orange


def load_masses(data_file: str) -> np.ndarray:
    masses = []
    with open(data_file, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            masses.append(float(parts[0]))
    return np.array(masses, dtype=np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: python view_snapshot_3d.py <positions.npy> <data_file>")
        sys.exit(1)

    snapshot_path = sys.argv[1]
    data_file = sys.argv[2]

    # Charge les positions du snapshot
    positions = np.load(snapshot_path)
    if positions.ndim != 2 or positions.shape[1] != 3:
        print("Le fichier de snapshot doit contenir un array de forme (N,3).")
        sys.exit(1)

    # Charge les masses pour reconstruire couleurs et intensités
    masses = load_masses(data_file)
    if masses.shape[0] != positions.shape[0]:
        print("Attention: nombre de masses et de positions différent !")
        print(f"masses: {masses.shape[0]}, positions: {positions.shape[0]}")
        # On s'arrête pour éviter un affichage incohérent
        sys.exit(1)

    max_mass = float(np.max(masses))
    colors = np.array([generate_star_color(float(m)) for m in masses], dtype=np.float32)
    intensity = np.clip(masses / max_mass, 0.5, 1.0)

    # Définition des bornes de la scène à partir des positions du snapshot
    mins = positions.min(axis=0) - 1.0e-6
    maxs = positions.max(axis=0) + 1.0e-6
    bounds = [[float(mins[0]), float(maxs[0])],
              [float(mins[1]), float(maxs[1])],
              [float(mins[2]), float(maxs[2])]]

    visu = visualizer3d.Visualizer3D(positions, colors, intensity, bounds)
    visu.run()


if __name__ == "__main__":
    main()
