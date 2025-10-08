import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# === Ignorar warnings ===
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Ajuste de rutas ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.runge_kutta_4_system import runge_kutta_4_system

# === Directorios base ===
base_dir = os.path.dirname(os.path.abspath(__file__))
rk4_dir = os.path.join(base_dir, "rungen_kutta4")
os.makedirs(rk4_dir, exist_ok=True)
os.makedirs(os.path.join(rk4_dir, "rungen_lorenz"), exist_ok=True)
os.makedirs(os.path.join(rk4_dir, "rungen_error/x"), exist_ok=True)
os.makedirs(os.path.join(rk4_dir, "rungen_error/y"), exist_ok=True)
os.makedirs(os.path.join(rk4_dir, "rungen_error/z"), exist_ok=True)

error_log_path = os.path.join(rk4_dir, "rungen_lorenz", "rungen_graficos_null.txt")
#############################################################################################
#############################################################################################

# Esta es una funcion generadora (closure) para crear la función f con el parámetro u fijo
def f_generator(u):
    def f_local(t, functions):
        x, y = functions
        return np.array([y,
                         u*(1-x**2)*y - x,])
    return f_local

# === Parámetros generales ===
u = [10,50,100,250]
y0_values = np.array([[2, 0],
                      [0, 1]])
h_values = [1e-2, 5e-3, 2e-3, 1e-3]


#############################################################################################
#############################################################################################

# === Carpeta con resultados de RKF45 ===
rkf45_folder = os.path.join(base_dir, "runge_kutta_45_resultados")
if not os.path.exists(rkf45_folder):
    print(f"[❌] No se encontró la carpeta RKF45 en: {rkf45_folder}")
    print("Asegúrate de haber ejecutado primero el script de Runge-Kutta-Fehlberg.")
else:
    print(f"[✅] Carpeta RKF45 encontrada: {rkf45_folder}")

#############################################################################################
#############################################################################################


# ======================================================
# =============== MÉTODO PRINCIPAL =====================
# ======================================================
def main_rk4_system(f_generator, y0, u, h):
    if u == 10:
        T = 50
    elif u >= 50:
        T = 200
    f = f_generator(u)

    """Ejecuta Runge-Kutta 4 y genera gráficos + comparación con RKF45"""
    t_vals, y_vals = runge_kutta_4_system(f, 0, y0, T, h)
    y_vals = np.array(y_vals)

    filename_base = f"Oscillator_h{h:.4f}_y0_x{y0[0]:.2f}_y0_y{y0[1]:.2f}_u{u:.2f}".replace('.', 'p')

    # === Gráfica del sistema ===
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[:, 0], label='X (RK4)', color="#e50eaf", linewidth=1)
    plt.plot(t_vals, y_vals[:, 1], label='Y (RK4)', color="#d589e1", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("Functions of the system")
    plt.legend()
    plt.title(f"Runge-Kutta 4 for Oscillator System\nh={h}, y0_x={y0[0]}, y0_y={y0[1]}, u={u}")
    plt.savefig(os.path.join(rk4_dir, "rungen_Oscillator", f"{filename_base}.png"))
    plt.close()


#############################################################################################
#############################################################################################

    # === Verificación de estabilidad ===
    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename_base}.png - contiene NaN o inf\n")
        return None, None, None

    return compare_with_rkf45(t_vals, y_vals, y0, h, filename_base)



# ======================================================
# =============== COMPARACIÓN CON RKF45 ================
# ======================================================
def compare_with_rkf45(t_rk4, y_rk4, y0, h, filename_base):
    """Compara RK4 contra RKF45 e imprime los errores"""
    # --- búsqueda flexible ---
    y0z_str = f"{y0[2]:.8f}".replace('.', 'p')
    csv_files = [f for f in os.listdir(rkf45_folder) if f.startswith(f"rkf45_init{y0z_str[:8]}")]

    if not csv_files:
        print(f"[⚠️] No se encontró archivo RKF45 para y0_z={y0[2]}")
        print(f"    (buscado patrón: rkf45_init{y0z_str[:8]})")
        return None, None, None

    rkf45_file = os.path.join(rkf45_folder, csv_files[0])
    df_rkf45 = pd.read_csv(rkf45_file)

    # --- Interpolación RKF45 a tiempos de RK4 ---
    x_interp = np.interp(t_rk4, df_rkf45["t"], df_rkf45["x"])
    y_interp = np.interp(t_rk4, df_rkf45["t"], df_rkf45["y"])
    z_interp = np.interp(t_rk4, df_rkf45["t"], df_rkf45["z"])

    # --- Errores absolutos ---
    err_x = np.abs(y_rk4[:, 0] - x_interp)
    err_y = np.abs(y_rk4[:, 1] - y_interp)
    err_z = np.abs(y_rk4[:, 2] - z_interp)

    # --- Graficar errores ---
    ejes_info = [
        (err_x, "X", "#e41aa1", "x"),
        (err_y, "Y", "#75b5e9", "y"),
        (err_z, "Z", "#ffbeef", "z")
    ]

    for err, eje, color, carpeta in ejes_info:
        plt.figure(figsize=(8, 5))
        plt.plot(t_rk4, err, color=color, linewidth=1.2)
        plt.title(f"Error absoluto |RK4 - RKF45| eje {eje}\n(h={h}, y0_z={y0[2]})")
        plt.xlabel("t")
        plt.ylabel(f"Error absoluto en {eje}")
        plt.yscale("log")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(rk4_dir, "rungen_error", carpeta, f"{filename_base}_error_{eje}.png"))
        plt.close()

    print(f"[✅] Gráficas de error generadas para h={h}, y0_z={y0[2]}")
    return err_x, err_y, err_z


# ======================================================
# =============== EJECUCIÓN PRINCIPAL ==================
# ======================================================
errors_df = pd.DataFrame(columns=["y0_z", "h", "norm_err_x", "norm_err_y", "norm_err_z"])

for y0 in y0_values:
    for h in h_values:
        print(f"Ejecutando RK4 con h={h}, y0_z={y0[2]}")
        err_x, err_y, err_z = main_rk4_system(f, y0, T, h)
        if err_x is None:
            continue

        norm_err_x = np.linalg.norm(err_x) / np.sqrt(len(err_x))
        norm_err_y = np.linalg.norm(err_y) / np.sqrt(len(err_y))
        norm_err_z = np.linalg.norm(err_z) / np.sqrt(len(err_z))

        errors_df.loc[len(errors_df)] = [y0[2], h, norm_err_x, norm_err_y, norm_err_z]

summary_path = os.path.join(rk4_dir, "rungen_kutta4_error_resumen.csv")
errors_df.to_csv(summary_path, index=False)
print(f"\n[📁] Resultados de errores guardados en {summary_path}")
print(errors_df)
