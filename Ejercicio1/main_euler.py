import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.euler_system import euler_system
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Ajuste de rutas ===
base_dir = os.path.dirname(os.path.abspath(__file__))
euler_dir = os.path.join(base_dir, "euler")
os.makedirs(euler_dir, exist_ok=True)
os.makedirs(os.path.join(euler_dir, "euler_lorenz"), exist_ok=True)
os.makedirs(os.path.join(euler_dir, "euler_error/x"), exist_ok=True)
os.makedirs(os.path.join(euler_dir, "euler_error/y"), exist_ok=True)
os.makedirs(os.path.join(euler_dir, "euler_error/z"), exist_ok=True)

error_log_path = os.path.join(euler_dir, "euler_lorenz", "euler_graficos_null.txt")

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
y0_values = np.array([
    [1, 1, 1],
    [1, 1, 1 + 10e-6],
    [1, 1, 1 + 10e-8]
])
h_values = [1e-2, 5e-3, 2e-3, 1e-3] 

# === Carpeta RKF45 ===
rkf45_folder = os.path.join(base_dir, "runge_kutta_45_resultados")
if not os.path.exists(rkf45_folder):
    print(f"[❌] No se encontró la carpeta RKF45 en: {rkf45_folder}")
    print("Asegúrate de haber ejecutado primero el script de Runge-Kutta-Fehlberg.")
else:
    print(f"[✅] Carpeta RKF45 encontrada: {rkf45_folder}")

# === Funciones ===
def main_euler_system(f, y0, T, h):
    t_vals, y_vals = euler_system(f, 0, y0, T, h)
    y_vals = np.array(y_vals)

    filename_base = f"lorenz_h{h:.4f}_init{y0[2]:.8f}".replace('.', 'p')

    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[:, 0], label='X (Euler)', color="#e50eaf", linewidth=1)
    plt.plot(t_vals, y_vals[:, 1], label='Y (Euler)', color="#d589e1", linewidth=1)
    plt.plot(t_vals, y_vals[:, 2], label='Z (Euler)', color="#f0a8cc", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("Functions of the system")
    plt.legend()
    plt.title(f"Euler Method for Lorenz System\nh={h}, y0_z={y0[2]}")
    plt.savefig(os.path.join(euler_dir, "euler_lorenz", f"{filename_base}.png"))
    plt.close()

    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename_base}.png - contiene NaN o inf\n")
        return None, None, None

    return compare_with_rkf45(t_vals, y_vals, y0, h, filename_base)


def compare_with_rkf45(t_euler, y_euler, y0, h, filename_base):
    y0z_str = f"{y0[2]:.8g}".replace('.', 'p')
    csv_files = [
    f for f in os.listdir(rkf45_folder)
    if f.startswith("rkf45_init") and f.endswith(".csv")
    and abs(float(f.split("init")[1].split("_")[0].replace('p', '.')) - y0[2]) < 1e-6
]


    if not csv_files:
        print(f"[⚠️] No se encontró archivo RKF45 para y0={y0[2]}")
        print(f"    (buscado patrón: rkf45_init{y0z_str})")
        return None, None, None

    rkf45_file = os.path.join(rkf45_folder, csv_files[0])
    df_rkf45 = pd.read_csv(rkf45_file)

    x_interp = np.interp(t_euler, df_rkf45["t"], df_rkf45["x"])
    y_interp = np.interp(t_euler, df_rkf45["t"], df_rkf45["y"])
    z_interp = np.interp(t_euler, df_rkf45["t"], df_rkf45["z"])

    err_x = np.abs(y_euler[:, 0] - x_interp)
    err_y = np.abs(y_euler[:, 1] - y_interp)
    err_z = np.abs(y_euler[:, 2] - z_interp)

    ejes_info = [
        (err_x, "X", "#e41aa1", "x"),
        (err_y, "Y", "#75b5e9", "y"),
        (err_z, "Z", "#ffbeef", "z")
    ]

    for err, eje, color, carpeta in ejes_info:
        plt.figure(figsize=(8, 5))
        plt.plot(t_euler, err, color=color, linewidth=1.2)
        plt.title(f"Error absoluto |Euler - RKF45| eje {eje}\n(h={h}, y0_z={y0[2]})")
        plt.xlabel("t")
        plt.ylabel(f"Error absoluto en {eje}")
        plt.yscale("log")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(euler_dir, "euler_error", carpeta, f"{filename_base}_error_{eje}.png"))
        plt.close()

    print(f"[✅] Gráficas de error generadas para h={h}, y0_z={y0[2]}")
    return err_x, err_y, err_z


# === Ejecución principal ===
errors_df = pd.DataFrame(columns=["y0_z", "h", "norm_err_x", "norm_err_y", "norm_err_z"])

for y0 in y0_values:
    for h in h_values:
        print(f"Ejecutando Euler con h={h}, y0_z={y0[2]}")
        err_x, err_y, err_z = main_euler_system(f, y0, T, h)
        if err_x is None:
            continue

        norm_err_x = np.linalg.norm(err_x) / np.sqrt(len(err_x))
        norm_err_y = np.linalg.norm(err_y) / np.sqrt(len(err_y))
        norm_err_z = np.linalg.norm(err_z) / np.sqrt(len(err_z))

        errors_df.loc[len(errors_df)] = [y0[2], h, norm_err_x, norm_err_y, norm_err_z]

summary_path = os.path.join(euler_dir, "euler_error_resumen.csv")
errors_df.to_csv(summary_path, index=False)
print(f"\n[📁] Resultados de errores guardados en {summary_path}")
print(errors_df)
