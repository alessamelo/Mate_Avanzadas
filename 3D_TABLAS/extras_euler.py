import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Ajuste de rutas ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.euler_system import euler_system

# === Directorio de salida único ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "euler", "extras_euler")
os.makedirs(output_dir, exist_ok=True)

# === Sistema de Lorenz ===
def f(t, functions):
    x, y, z = functions
    sigma = 10
    rho = 28
    beta = 8 / 3
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

# === Parámetros generales ===
T = 50
h = 2e-3
y0_values = np.array([
    [1, 1, 1],
    [1, 1, 1 + 10e-6],
    [1, 1, 1 + 10e-8]
])
colors = ["#0077FF", "#FF6600", "#00FF66"]

# === Ejecución principal ===
for i, y0 in enumerate(y0_values):
    t_vals, y_vals = euler_system(f, 0, y0, T, h)
    y_vals = np.array(y_vals)
    x, y_, z = y_vals[:, 0], y_vals[:, 1], y_vals[:, 2]
    label = f"y0_z={y0[2]:.8f}"
    filename_base = f"y0_{str(y0[2]).replace('.', 'p')}"

    # === Gráfica 3D individual ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y_, z, color=colors[i], linewidth=0.8)
    ax.set_title(f"Lorenz 3D — {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lorenz_3D_{filename_base}.png"))
    plt.close()

    # === Sección de Poincaré individual ===
    poincare_points = []
    for j in range(len(x) - 1):
        if x[j] < 0 and x[j+1] > 0 and y_[j] > 0:
            alpha = -x[j] / (x[j+1] - x[j])
            y_cross = y_[j] + alpha * (y_[j+1] - y_[j])
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

print("[✅] 6 Euler graphics successfully saved in:", output_dir)
