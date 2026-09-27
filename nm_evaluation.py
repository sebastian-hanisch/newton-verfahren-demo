"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Ein-Schritt-Korrektheits-Kette,
Reduktions-Check auf Gradientenabstieg, quadratische Konvergenzrate nahe dem Optimum,
Divergenzschwelle der logarithmischen Beule, Rosenbrock-Konvergenz-ohne-Monotonie-Befund,
Gradienten-/Hesse-Check."""
from dataclasses import dataclass

import numpy as np

import nm_functions as fn
import nm_optimizer as opt


@dataclass(frozen=True)
class Settings:
    func: str
    x0: float  # bei Quadratik ungenutzt (x0 kommt aus dem Seed); bei Rosenbrock Diagonalpunkt
              # (x0, x0); bei Log-Beule der 1D-Startwert
    max_iter: int
    seed: int
    condition_number: float = 20.0


def analyse(settings: Settings) -> dict:
    if settings.func == "quadratic":
        A = fn.random_spd_matrix(2, settings.condition_number, settings.seed)
        f, grad, hess = fn.quadratic(A)
        x_star = np.zeros(2)
        rng = np.random.default_rng(settings.seed + 1)
        x0 = rng.normal(size=2)
    elif settings.func == "rosenbrock":
        f, grad, hess = fn.rosenbrock()
        x_star = np.array([1.0, 1.0])
        x0 = np.array([settings.x0, settings.x0])
    else:
        f, grad, hess = fn.log_bump()
        x_star = np.zeros(1)
        x0 = np.array([settings.x0])
    result = opt.newton_method(f, grad, hess, x0, max_iter=settings.max_iter)
    return {"result": result, "f_star": 0.0, "x_star": x_star, "x0": x0}


def exact_one_step_check(kappas=(2, 20, 200), dim=5, n_trials=3) -> list:
    """Newton loest die Quadratik in GENAU einem Schritt, unabhaengig von der Konditionszahl."""
    rows = []
    for kappa in kappas:
        for trial in range(n_trials):
            A = fn.random_spd_matrix(dim, kappa, seed=trial)
            f, grad, hess = fn.quadratic(A)
            rng = np.random.default_rng(trial + 100)
            x0 = rng.normal(size=dim)
            result = opt.newton_method(f, grad, hess, x0, max_iter=5)
            rows.append({"kappa": kappa, "trial": trial, "n_iter": result.n_iter,
                        "f_final": result.fvals[-1], "x_final_norm": float(np.linalg.norm(result.trajectory[-1]))})
    return rows


def reduction_to_gradient_descent_check(dim=4, condition_number=10.0, seed=0, eta=0.05,
                                        max_iter=20) -> float:
    """Ersetzt man die Hesse-Matrix durch (1/eta) I, wird Newton algebraisch identisch zum
    Gradientenabstieg-Update aus Stueck 1."""
    A = fn.random_spd_matrix(dim, condition_number, seed)
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=dim)

    def hess_fake(x):
        return np.eye(dim) / eta

    r_newton_fake = opt.newton_method(f, grad, hess_fake, x0, max_iter=max_iter)
    r_gd = opt.gradient_descent(f, grad, x0, eta, max_iter=max_iter)
    n = min(len(r_newton_fake.trajectory), len(r_gd.trajectory))
    return float(np.max(np.abs(r_newton_fake.trajectory[:n] - r_gd.trajectory[:n])))


def quadratic_convergence_exponents(starts=((0.9, 0.8), (1.1, 1.3), (0.5, 0.2)), max_iter=15) -> list:
    """Fehler-Verhaeltnis log(e_{k+1})/log(e_k) sollte sich 2 annaehern (quadratische Konvergenz),
    sobald der Fehler klein genug ist."""
    f, grad, hess = fn.rosenbrock()
    x_star = np.array([1.0, 1.0])
    rows = []
    for x0 in starts:
        result = opt.newton_method(f, grad, hess, np.array(x0), max_iter=max_iter)
        errs = np.linalg.norm(result.trajectory - x_star, axis=1)
        logs = np.log(np.maximum(errs, 1e-300))
        exponents = []
        for i in range(len(logs) - 1):
            if 1e-14 < errs[i] < 0.5 and errs[i + 1] > 1e-14 and logs[i] != 0:
                exponents.append(float(logs[i + 1] / logs[i]))
        rows.append({"x0": x0, "errors": errs.tolist(), "exponents": exponents,
                    "last_exponent": exponents[-1] if exponents else float("nan")})
    return rows


def rosenbrock_wide_start_sweep(n_random=30, radius=2.0, seed=7) -> dict:
    """Ehrlicher Befund (Plan-Korrektur): volles, ungesichertes Newton konvergiert auf Rosenbrock
    von JEDEM getesteten Startpunkt (auch weit entfernt), ABER ohne garantierte monotone Abnahme
    von f entlang des Weges."""
    f, grad, hess = fn.rosenbrock()
    x_star = np.array([1.0, 1.0])
    rng = np.random.default_rng(seed)
    starts = [rng.uniform(-radius, radius, size=2) for _ in range(n_random)]
    starts += [np.array([-1.2, 1.0]), np.array([-1.5, -1.0]), np.array([2.0, -1.0]),
               np.array([-2.0, 2.0])]
    n_converged = 0
    n_non_monotone = 0
    max_iters_used = 0
    for x0 in starts:
        result = opt.newton_method(f, grad, hess, x0, max_iter=50)
        final_err = float(np.linalg.norm(result.trajectory[-1] - x_star))
        if final_err < 1e-6:
            n_converged += 1
        max_iters_used = max(max_iters_used, result.n_iter)
        fvals = result.fvals
        if any(fvals[i + 1] > fvals[i] for i in range(len(fvals) - 1)):
            n_non_monotone += 1
    return {"n_total": len(starts), "n_converged": n_converged,
            "n_non_monotone": n_non_monotone, "max_iters_used": max_iters_used}


def log_bump_divergence_threshold(deltas=(-0.01, -0.001, -0.0001, 0.0001, 0.001, 0.01)) -> dict:
    """Exakte Divergenzschwelle |x0| = 1/sqrt(3), von Hand herleitbar aus der Fixpunkt-Iteration
    x_{k+1} = 2 x_k^3 / (x_k^2 - 1)."""
    threshold = 1.0 / np.sqrt(3.0)
    f, grad, hess = fn.log_bump()
    rows = []
    for delta in deltas:
        x0 = threshold + delta
        result = opt.newton_method(f, grad, hess, np.array([x0]), max_iter=60)
        rows.append({"delta": delta, "x0": x0, "diverged": bool(result.diverged),
                    "n_iter": result.n_iter})
    return {"theoretical_threshold": threshold, "rows": rows}


def gradient_check(eps: float = 1e-6) -> dict:
    rng = np.random.default_rng(0)
    A = fn.random_spd_matrix(6, 10.0, seed=1)
    f, grad, hess = fn.quadratic(A)
    x = rng.normal(size=6)
    analytic = grad(x)
    numeric = np.zeros(6)
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    quad_err = float(np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))

    f2, grad2, hess2 = fn.rosenbrock()
    x2 = np.array([0.3, -0.7])
    analytic2 = grad2(x2)
    numeric2 = np.zeros(2)
    for i in range(2):
        xp, xm = x2.copy(), x2.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric2[i] = (f2(xp) - f2(xm)) / (2 * eps)
    rosen_err = float(np.max(np.abs(analytic2 - numeric2) / np.maximum(np.abs(analytic2), 1e-8)))
    return {"quadratic_grad_max_rel_err": quad_err, "rosenbrock_grad_max_rel_err": rosen_err}


def _numeric_hessian(grad, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        H[:, i] = (grad(xp) - grad(xm)) / (2 * eps)
    return H


def hessian_check() -> dict:
    A = fn.random_spd_matrix(5, 10.0, seed=3)
    f, grad, hess = fn.quadratic(A)
    x = np.random.default_rng(0).normal(size=5)
    quad_err = float(np.max(np.abs(hess(x) - _numeric_hessian(grad, x))))

    f2, grad2, hess2 = fn.rosenbrock()
    x2 = np.array([0.3, -0.7])
    analytic2 = hess2(x2)
    numeric2 = _numeric_hessian(grad2, x2)
    rosen_err = float(np.max(np.abs(analytic2 - numeric2) / np.maximum(np.abs(analytic2), 1e-8)))
    return {"quadratic_hess_max_abs_err": quad_err, "rosenbrock_hess_max_rel_err": rosen_err}
