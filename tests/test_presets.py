import numpy as np

import nm_constants as C
import nm_functions as fn
import nm_optimizer as opt


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["func"] in C.FUNCTIONS


def test_preset_quadratik_ein_schritt_converges_in_one_step():
    p = C.PRESETS["quadratik_ein_schritt"]
    A = fn.random_spd_matrix(C.DIM, p["kappa"], p["seed"])
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(p["seed"] + 1)
    x0 = rng.normal(size=C.DIM)
    result = opt.newton_method(f, grad, hess, x0, max_iter=p["max_iter"])
    assert result.n_iter == 1


def test_preset_rosenbrock_nah_converges_quadratically():
    p = C.PRESETS["rosenbrock_nah"]
    f, grad, hess = fn.rosenbrock()
    x0 = np.array([p["x0"], p["x0"]])
    result = opt.newton_method(f, grad, hess, x0, max_iter=p["max_iter"])
    assert result.converged


def test_preset_log_beule_konvergiert_stays_finite():
    p = C.PRESETS["log_beule_konvergiert"]
    f, grad, hess = fn.log_bump()
    result = opt.newton_method(f, grad, hess, np.array([p["x0"]]), max_iter=p["max_iter"])
    assert result.converged
    assert not result.diverged


def test_preset_log_beule_divergiert_actually_diverges():
    p = C.PRESETS["log_beule_divergiert"]
    f, grad, hess = fn.log_bump()
    result = opt.newton_method(f, grad, hess, np.array([p["x0"]]), max_iter=p["max_iter"])
    assert result.diverged
