import numpy as np

def runge_kutta_4_system(f, t0, y0, T, h):
    """
    Método de Runge-Kutta de orden 4 para resolver un sistema de EDOs: dy/dt = f(t, y)

    Parámetros:
    f  : función f(t, y) que retorna un vector (np.array)
    t0 : valor inicial del tiempo
    y0 : condición inicial como vector (np.array o lista)
    h  : tamaño de paso
    n  : número de pasos

    Retorna:
    t_vals : arreglo de tiempos
    y_vals : arreglo de soluciones (n+1, len(y0))
    """
    n = int((T - t0) / h) -1  # number of points in (t0,T)
    y0 = np.array(y0, dtype=float)  # aseguramos que y0 es array
    m = len(y0)                     # número de ecuaciones

    t_vals = np.zeros(n+2)
    y_vals = np.zeros((n+2, m))

    t_vals[0] = t0
    y_vals[0] = y0

    for i in range(n+1):
        t = t_vals[i]
        y = y_vals[i]

        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h,   y + h   * k3)

        y_vals[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_vals[i+1] = t + h

    return t_vals, y_vals
