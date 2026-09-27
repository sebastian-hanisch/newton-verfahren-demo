"""Korrektheits-/Verhaltenstests mit billigen Parametern. Die offiziellen, teuren Sweep-Werte
stehen mit Toleranzband in test_claims.py (Modul-Fixtures, je Sweep nur einmal berechnet)."""
import nm_evaluation as ev


def test_exact_one_step_check_runs_with_small_values():
    rows = ev.exact_one_step_check(kappas=(5,), n_trials=1)
    assert len(rows) == 1
    assert rows[0]["n_iter"] == 1


def test_reduction_check_is_near_machine_precision():
    assert ev.reduction_to_gradient_descent_check(max_iter=5) < 1e-10


def test_log_bump_divergence_threshold_has_correct_sign():
    out = ev.log_bump_divergence_threshold(deltas=(-0.01, 0.01))
    assert not out["rows"][0]["diverged"]
    assert out["rows"][1]["diverged"]


def test_analyse_returns_valid_result_for_each_function():
    for func in ("quadratic", "rosenbrock", "log_bump"):
        settings = ev.Settings(func=func, x0=0.9, max_iter=10, seed=0)
        out = ev.analyse(settings)
        assert out["result"].fvals[0] >= 0.0
