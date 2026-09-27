"""Newton-Verfahren (voll, ungesichert - keine Liniensuche/Trust-Region) und eine eigenstaendige
Kopie des Gradientenabstiegs aus Stueck 1, nur fuer den Reduktions-Korrektheits-Check."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    trajectory: np.ndarray
    fvals: np.ndarray
    converged: bool
    n_iter: int
    diverged: bool = False


def newton_method(f, grad, hess, x0, max_iter=50, tol=1e-12) -> Result:
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    traj = [x.copy()]
    fvals = [float(f(x))]
    converged = False
    diverged = False
    k = 0
    for k in range(1, max_iter + 1):
        g = grad(x)
        if float(g @ g) < tol ** 2:
            converged = True
            break
        H = hess(x)
        try:
            step = np.linalg.solve(np.atleast_2d(H), g)
        except np.linalg.LinAlgError:
            diverged = True
            break
        x = x - step
        traj.append(x.copy())
        if not np.all(np.isfinite(x)):
            fvals.append(float("inf"))
            diverged = True
            break
        fvals.append(float(f(x)))
        if fvals[-1] > 1e12 or np.any(np.abs(x) > 1e6):
            diverged = True
            break
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1, diverged=diverged)


def gradient_descent(f, grad, x0, eta, max_iter=1000, tol=1e-10) -> Result:
    """Eigenstaendige Kopie aus gradient-descent-demo, nur fuer den Reduktions-Check hier."""
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    traj = [x.copy()]
    fvals = [float(f(x))]
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        g = grad(x)
        if float(g @ g) < tol ** 2:
            converged = True
            break
        x = x - eta * g
        traj.append(x.copy())
        fvals.append(float(f(x)))
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1)
