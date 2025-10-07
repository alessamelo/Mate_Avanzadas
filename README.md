# 🧮 Proyecto Métodos Numéricos — Ejercicio 1

Amixs 💻  
Verán, dentro de la carpeta **`Ejercicio1`** ya están actualizados todos los códigos para el proyecto.  
También subí los **CSV que yo generé**, así que pueden usarlos directamente si no quieren volver a correr todo desde cero (porque el RKF45 demora full 😭).

---

## 📂 Archivos principales

| Archivo | Descripción |
|----------|--------------|
| `main_euler.py` | Implementa el método de **Euler explícito** para el sistema de Lorenz. Genera las gráficas del sistema y los errores comparados con RKF45. |
| `main_rk.py` | Implementa el **Runge-Kutta de 4° orden (RK4)** para el mismo sistema. También genera los errores respecto al RKF45. |
| `main_rk45.py` | Usa el **Runge-Kutta-Fehlberg adaptativo (RKF45)** como método de referencia, con paso variable. Es el más preciso y genera los CSV base para comparar. |

---

## 🧠 Qué deben hacer

1. Asegúrense de tener instalada la carpeta completa **`Ejercicio1`** con todas las subcarpetas.
2. Los archivos `.csv` del RKF45 ya están subidos (por si no quieren recalcularlos).
3. Corran los siguientes scripts en orden (si desean regenerar todo desde cero):

   ```bash
   python main_rk45.py   # Genera los CSV de referencia
   python main_euler.py  # Compara Euler con RKF45
   python main_rk.py     # Compara RK4 con RKF45
