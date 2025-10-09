import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Ajuste de rutas ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.runge_kutta_45_system import runge_kutta_45_system

# === Directorio de salida ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "rkf45_lorenz", "sensibilidad")
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
tol = 1e-9
t_targets = [10, 20, 30, 40, 50]

# === Condiciones iniciales ===
y0_values = np.array([
    [1, 1, 1],
    [1, 1, 1 + 10e-6],
    [1, 1, 1 + 10e-8]
])

# === Simulación base ===
t_base, y_base = runge_kutta_45_system(f, y0=y0_values[0], T=T, tol=tol)
t_base = np.array(t_base)
y_base = np.array(y_base)

# === Cálculo de sensibilidad y NFE ===
columns = ["y0_z", "tol", "NFE"] + [f"Δ({t})" for t in t_targets]
sensitivity_df = pd.DataFrame(columns=columns)

for y0 in y0_values:
    t_vals, y_vals = runge_kutta_45_system(f, y0=y0, T=T, tol=tol)
    t_vals = np.array(t_vals)
    y_vals = np.array(y_vals)

    deltas = []
    for t_target in t_targets:
        idx_base = np.argmin(np.abs(t_base - t_target))
        idx_curr = np.argmin(np.abs(t_vals - t_target))
        is_base = np.allclose(y0, y0_values[0], rtol=0, atol=1e-12)
        delta_t = 0.0 if is_base else np.linalg.norm(y_vals[idx_curr] - y_base[idx_base])
        deltas.append(delta_t)

    nfe = len(t_vals) * 6  # RKF45: 6 evaluaciones por paso
    sensitivity_df.loc[len(sensitivity_df)] = [y0[2], tol, nfe] + deltas

# === Guardar CSV ===
csv_path = os.path.join(output_dir, "rkf45_sensibilidad_con_NFE.csv")
sensitivity_df.to_csv(csv_path, index=False)
print(f"\n[✅] Sensibilidad Δ(t) + NFE guardada en: {csv_path}")
print(sensitivity_df)
