import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Directorios ===
base_dir = os.path.dirname(os.path.abspath(__file__))
rkf45_folder = os.path.join(base_dir, "runge_kutta_45_resultados")
output_dir = os.path.join(rkf45_folder, "extras")
os.makedirs(output_dir, exist_ok=True)

# === Condiciones iniciales esperadas ===
y0_values = [1, 1 + 10e-6, 1 + 10e-8]
colors = ["#0077FF", "#FF6600", "#00FF66"]

# === Procesar cada archivo RKF45 ===
for i, y0_z in enumerate(y0_values):
    y0z_str = f"{y0_z:.8f}".replace('.', 'p')
    csv_files = [f for f in os.listdir(rkf45_folder) if f.startswith(f"rkf45_init{y0z_str[:8]}")]

    if not csv_files:
        print(f"[⚠️] RKF45 file not found for y0_z={y0_z}")
        continue

    rkf45_file = os.path.join(rkf45_folder, csv_files[0])
    df = pd.read_csv(rkf45_file)

    # === Extraer tolerancia del nombre del archivo ===
    filename_parts = csv_files[0].split("_tol")
    tolerance_str = filename_parts[1].replace(".csv", "") if len(filename_parts) > 1 else "unknown"

    x, y, z = df["x"].values, df["y"].values, df["z"].values
    label = f"y0_z={y0_z:.8f}, tol={tolerance_str}"
    filename_base = f"y0_{str(y0_z).replace('.', 'p')}_tol{tolerance_str}"

    # === Gráfica 3D ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color=colors[i], linewidth=0.8)
    ax.set_title(f"Lorenz 3D — {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lorenz_3D_{filename_base}.png"))
    plt.close()

    # === Sección de Poincaré ===
    poincare_points = []
    for j in range(len(x) - 1):
        if x[j] < 0 and x[j+1] > 0 and y[j] > 0:
            alpha = -x[j] / (x[j+1] - x[j])
            y_cross = y[j] + alpha * (y[j+1] - y[j])
            z_cross = z[j] + alpha * (z[j+1] - z[j])
            poincare_points.append((y_cross, z_cross))
    poincare_points = np.array(poincare_points)

    plt.figure(figsize=(8, 6))
    plt.scatter(poincare_points[:, 0], poincare_points[:, 1], s=10, color=colors[i], alpha=0.6)
    plt.title(f"Poincaré Section — {label}")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"poincare_{filename_base}.png"))
    plt.close()

print("[✅] RKF45 3D and Poincaré graphics saved in:", output_dir)
