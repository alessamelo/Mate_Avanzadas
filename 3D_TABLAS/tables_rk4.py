import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Ajuste de rutas ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.runge_kutta_4_system import runge_kutta_4_system

# === Directorio de salida ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "rungen_kutta4", "sensibilidad")
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

# === Parámetros ===
T = 50
t_targets = [10, 20, 30, 40, 50]

# === Condiciones específicas ===
condiciones = [
    {"y0": [1, 1, 1], "h": 1e-3},
    {"y0": [1, 1, 1 + 10e-6], "h": 1e-3},
    {"y0": [1, 1, 1 + 10e-8], "h": 2e-3}
]

# === Simulación base ===
base_cond = condiciones[0]
t_base, y_base = runge_kutta_4_system(f, 0, base_cond["y0"], T, base_cond["h"])
y_base = np.array(y_base)

# === Cálculo de sensibilidad y NFE ===
columns = ["y0_z", "h", "NFE"] + [f"Δ({t})" for t in t_targets]
sensitivity_df = pd.DataFrame(columns=columns)

for cond in condiciones:
    y0 = cond["y0"]
    h = cond["h"]
    t_vals, y_vals = runge_kutta_4_system(f, 0, y0, T, h)
    y_vals = np.array(y_vals)

    deltas = []
    for t_target in t_targets:
        idx_base = np.argmin(np.abs(t_base - t_target))
        idx_curr = np.argmin(np.abs(t_vals - t_target))
        delta_t = np.linalg.norm(y_vals[idx_curr] - y_base[idx_base]) if y0 != base_cond["y0"] else 0.0
        deltas.append(delta_t)

    nfe = int(T / h) * 4
    sensitivity_df.loc[len(sensitivity_df)] = [y0[2], h, nfe] + deltas

# === Guardar CSV ===
csv_path = os.path.join(output_dir, "rk4_sensibilidad_con_NFE.csv")
sensitivity_df.to_csv(csv_path, index=False)
print(f"\n[✅] Sensibilidad Δ(t) + NFE guardada en: {csv_path}")
print(sensitivity_df)
