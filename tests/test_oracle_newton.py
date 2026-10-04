"""Unabhaengige Orakel-Tests (anderer Rechenweg als der Code):
- Rosenbrock-Newton: handgeschriebene 2x2-Iteration mit Cramerscher Regel (statt np.linalg.solve),
  Pfad und Schrittzahl; Funktionen/Gradient/Hesse gegen scipy.optimize.rosen*.
- Log-Beule: skalare Fixpunkt-Iteration x -> 2x^3/(x^2-1) und die Schwelle |x0| = 1/sqrt(3).
- Quadratik: ein Schritt gegen A^{-1}-freie Handrechnung (x1 = 0), unabhaengig von kappa."""
import math

import numpy as np
import pytest

import nm_functions as fn
import nm_optimizer as opt

scipy_opt = pytest.importorskip("scipy.optimize")


def _newton_rosenbrock_cramer(x0, max_iter=50, tol=1e-12):
    x, y = float(x0[0]), float(x0[1])
    path = [(x, y)]
    for _ in range(max_iter):
        gx = -2 * (1 - x) - 400 * x * (y - x * x)
        gy = 200 * (y - x * x)
        if gx * gx + gy * gy < tol ** 2:
            break
        a, b, d = 2 - 400 * (y - x * x) + 800 * x * x, -400 * x, 200.0
        det = a * d - b * b
        x, y = x - (gx * d - b * gy) / det, y - (a * gy - b * gx) / det
        path.append((x, y))
        if abs(x) > 1e6 or abs(y) > 1e6:
            break
    return np.array(path)


def test_rosenbrock_functions_match_scipy():
    rng = np.random.default_rng(1)
    f, g, h = fn.rosenbrock()
    for _ in range(100):
        x = rng.normal(size=2) * 2
        assert f(x) == pytest.approx(scipy_opt.rosen(x))
        np.testing.assert_allclose(g(x), scipy_opt.rosen_der(x))
        np.testing.assert_allclose(h(x), scipy_opt.rosen_hess(x))


def test_newton_path_matches_cramer_iteration_on_rosenbrock():
    rng = np.random.default_rng(2)
    f, g, h = fn.rosenbrock()
    for _ in range(120):
        x0 = rng.uniform(-2, 2, size=2)
        r = opt.newton_method(f, g, h, x0, max_iter=50)
        ref = _newton_rosenbrock_cramer(x0)
        assert r.n_iter == len(r.trajectory) - 1
        assert abs(len(ref) - len(r.trajectory)) <= 1
        m = min(len(ref), len(r.trajectory))
        np.testing.assert_allclose(r.trajectory[:m], ref[:m], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(r.trajectory[-1], [1.0, 1.0], atol=1e-6)


def test_newton_solves_quadratic_in_one_step_for_any_kappa():
    rng = np.random.default_rng(3)
    for _ in range(100):
        d = int(rng.integers(1, 8))
        kappa = float(np.exp(rng.uniform(0, np.log(500))))
        A = fn.random_spd_matrix(d, kappa, int(rng.integers(0, 10**6)))
        f, g, h = fn.quadratic(A)
        r = opt.newton_method(f, g, h, rng.normal(size=d), max_iter=5)
        assert r.n_iter == 1 and r.converged
        assert np.linalg.norm(r.trajectory[1]) < 1e-8


def _log_bump_iteration(x, max_iter=60):
    path = [x]
    for _ in range(max_iter):
        if (2 * x / (1 + x * x)) ** 2 < 1e-24:
            break
        x = 2 * x ** 3 / (x * x - 1)
        path.append(x)
        if abs(x) > 1e6:
            break
    return path


def test_log_bump_threshold_and_path_match_fixed_point_iteration():
    thr = 1 / math.sqrt(3)
    f, g, h = fn.log_bump()
    rng = np.random.default_rng(4)
    for x0 in rng.uniform(-3, 3, size=150):
        if abs(abs(x0) - thr) < 5e-4:
            continue
        r = opt.newton_method(f, g, h, np.array([x0]), max_iter=60)
        assert r.diverged == (abs(x0) > thr)
        ref = _log_bump_iteration(float(x0))
        m = min(len(ref), len(r.trajectory))
        np.testing.assert_allclose(r.trajectory[:m, 0], ref[:m], rtol=1e-6, atol=1e-9)


def test_log_bump_step_counts_just_below_threshold():
    """Handrechnung per Fixpunkt-Iteration: Schritte bis ||grad|| < 1e-12 knapp unter der Schwelle."""
    thr = 1 / math.sqrt(3)
    f, g, h = fn.log_bump()
    for delta in (-0.01, -0.001, -0.0001):
        r = opt.newton_method(f, g, h, np.array([thr + delta]), max_iter=60)
        assert r.converged and not r.diverged
        assert r.n_iter == len(_log_bump_iteration(thr + delta)) - 1
