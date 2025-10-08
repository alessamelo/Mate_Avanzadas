import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

# === Ajuste de rutas ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Metodos.runge_kutta_45_system import runge_kutta_45_system  # usa tu versión

# === Ruta base (carpeta donde está este archivo .py) ===
base_dir = os.path.dirname(os.path.abspath(__file__))

# === Crear subcarpetas dentro de la carpeta actual ===
rkf45_dir = os.path.join(base_dir, "rkf45_Oscillator")
rkf45_results_dir = os.path.join(base_dir, "runge_kutta_45_resultados")
os.makedirs(rkf45_dir, exist_ok=True)
os.makedirs(rkf45_results_dir, exist_ok=True)
error_log_path = os.path.join(rkf45_dir, "rkf45_graficos_null.txt")
 
# Esta es una funcion generadora (closure) para crear la función f con el parámetro u fijo
def f_generator(u):
    def f_local(t, functions):
        x, y = functions
        return np.array([y,
                         u*(1-x**2)*y - x,])
    return f_local

# === Parámetros generales ===
u = [10,50,100,250]
tol = 1e-7  # tolerancia fija
y0_values = np.array([[2, 0],
                      [0, 1]])

# === Función principal ===
def main_rkf45_system(f_generator, u, y0, tol=1e-10):
    if u == 10:
        T = 50
    elif u >= 50:
        T = 200
    f = f_generator(u)

    start_time = time.time()

    # Llamada al método RKF45
    t_vals, y_vals = runge_kutta_45_system(f, y0=y0, T=T, tol=tol)
    y_vals = np.array(y_vals)

    filename = f"rkf45_u{u:.2f}_y0_x{y0[0]:.2f}_y0_y{y0[1]:.2f}_tol{tol:.8f}".replace('.', 'p')

    # === Comprobación de NaN o inf ===
    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename}.png - contiene NaN o inf\n")
        print(f"[❌] {filename}: contiene NaN o inf")
        return None, None

    # === Guardar resultados (CSV) ===
    csv_path = os.path.join(rkf45_results_dir, f"{filename}.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["t", "x", "y", "T", "tol", "y0_x", "y0_y"])
        for ti, yi in zip(t_vals, y_vals):
            writer.writerow([ti, yi[0], yi[1], T, tol, y0[0], y0[1]])

    elapsed = time.time() - start_time
    print(f"[✅] {filename} guardado en {elapsed:.2f}s con {len(t_vals)} pasos.")

    # === Graficar ===
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[:, 0], label='X', color='#e41a1c', linewidth=1)
    plt.plot(t_vals, y_vals[:, 1], label='Y', color='#377eb8', linewidth=1)
    plt.xlabel("t")
    plt.ylabel("Functions of the system")
    plt.legend()

    param_text = f"u={u},y0_x={y0[0]},y0_y={y0[1]}"
    plt.title(f"RKF45 Oscillator System ({param_text})")
    plt.text(0.02, 0.02, param_text, transform=plt.gca().transAxes,
             fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.savefig(os.path.join(rkf45_dir, f"{filename}.png"))
    plt.close()

    return t_vals, y_vals


# === Ejecutar las ocho simulaciones ===
for u_i in u:
    for y0 in y0_values:
        main_rkf45_system(f_generator, u_i, y0, tol)
