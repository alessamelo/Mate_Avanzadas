import numpy as np

def euler_system(f, t0, y0, T, h):
    """
    Solves a system of ODEs using the Euler method.
    
    Parameters:
    - f : function, the RHS of the system dy/dt = f(t, y), returns a numpy array
    - t0 : float, initial time
    - y0 : array_like, initial condition (vector)
    - T : float, final time
    - h : float, step size

    Returns:
    - t_vals : array of time points
    - y_vals : array of y values at each time (shape: [n_steps+1, len(y0)])
    """
    N = int((T - t0) / h)
    t_vals = t0 + h * np.arange(N + 1)
    y_vals = np.zeros((N + 1, len(y0)))
    y_vals[0] = y0

    for i in range(1, N + 1):
        t = t_vals[i - 1]
        y = y_vals[i - 1]
        y_vals[i] = y + h * f(t, y)

    return t_vals, y_vals
