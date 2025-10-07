import numpy as np

def runge_kutta_45_system(f, y0, T, tol=1e-10, h_max=0.1, h_min=1e-12, t0=0):
    """
    Método de Runge-Kutta-Fehlberg 4(5) para sistemas de EDOs con paso adaptativo.

    Parámetros:
    f    : función f(t, y)
    t0   : tiempo inicial
    y0   : condición inicial (np.array)
    T    : tiempo final
    h0   : paso inicial
    TOL  : tolerancia
    h_max: paso máximo
    h_min: paso mínimo

    Retorna:
    t_vals : arreglo de tiempos
    y_vals : arreglo de soluciones
    """
        
    t_vals = [t0]
    y_vals = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)
    h = h_max
    while t < T:
        if t + h > T:
            h = T - t

        # Etapas RKF45
        K1 = h * f(t, y)
        K2 = h * f(t + 0.25*h, y + 0.25*K1)
        K3 = h * f(t + 0.375*h, y + (3/32)*K1 + (9/32)*K2)
        K4 = h * f(t + (12/13)*h, y + (1932/2197)*K1 - (7200/2197)*K2 + (7296/2197)*K3)
        K5 = h * f(t + h, y + (439/216)*K1 - 8*K2 + (3680/513)*K3 - (845/4104)*K4)
        K6 = h * f(t + 0.5*h, y - (8/27)*K1 + 2*K2 - (3544/2565)*K3 + (1859/4104)*K4 - (11/40)*K5)

        y4 = y + (25/216)*K1 + (1408/2565)*K3 + (2197/4104)*K4 - (1/5)*K5
        y5 = y + (16/135)*K1 + (6656/12825)*K3 + (28561/56430)*K4 - (9/50)*K5 + (2/55)*K6

        # Estimación del error
        R = np.max(np.abs((y5 - y4) / h))

        if R <= tol:
            t += h
            y = y5
            t_vals.append(t)
            y_vals.append(y.copy())

        # Ajuste del paso
        delta = 0.84 * (tol / R)**0.25 if R > 0 else 2.0
        h = min(h_max, max(h_min, delta * h))

        if h < h_min:
            print("Error: paso mínimo alcanzado.")
            break

    return t_vals, y_vals