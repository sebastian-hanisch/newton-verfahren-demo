"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet.
Modul-Fixtures berechnen jeden Sweep nur einmal; Toleranzband statt exakter Gleichheit, da einige
Groessen (Konvergenz-Exponenten) durch die numerische Natur der Iteration leicht schwanken."""
import pytest

import nm_evaluation as ev


@pytest.fixture(scope="module")
def one_step_rows():
    return ev.exact_one_step_check()


@pytest.fixture(scope="module")
def convergence_exponent_rows():
    return ev.quadratic_convergence_exponents()


@pytest.fixture(scope="module")
def wide_sweep():
    return ev.rosenbrock_wide_start_sweep()


@pytest.fixture(scope="module")
def divergence_data():
    return ev.log_bump_divergence_threshold()


def test_claim_newton_solves_quadratic_in_exactly_one_step(one_step_rows):
    """Unabhaengig von der Konditionszahl - im Gegensatz zu Gradientenabstieg (Stueck 1), das bei
    grossem kappa hunderte Schritte braucht."""
    for row in one_step_rows:
        assert row["n_iter"] == 1
        assert row["f_final"] < 1e-20


def test_claim_reduction_to_gradient_descent_is_near_machine_precision():
    assert ev.reduction_to_gradient_descent_check() < 1e-10


def test_claim_convergence_exponent_approaches_two_near_optimum(convergence_exponent_rows):
    """Quadratische Konvergenz: der Exponent naehert sich 2 (Fehler ~ Fehler^2), nicht 1 (linear)."""
    for row in convergence_exponent_rows:
        assert 1.5 < row["last_exponent"] < 2.5


def test_claim_rosenbrock_wide_start_always_converges_but_never_monotone(wide_sweep):
    """Echte Plan-Korrektur: die urspruengliche Hypothese (Newton scheitert fern vom Optimum auf
    Rosenbrock) wurde widerlegt - es konvergiert IMMER, aber nie monoton."""
    assert wide_sweep["n_converged"] == wide_sweep["n_total"]
    assert wide_sweep["n_non_monotone"] == wide_sweep["n_total"]
    assert wide_sweep["max_iters_used"] < 20


def test_claim_log_bump_threshold_is_sharp(divergence_data):
    for row in divergence_data["rows"]:
        if row["x0"] < divergence_data["theoretical_threshold"]:
            assert not row["diverged"]
        else:
            assert row["diverged"]


def test_claim_gradient_and_hessian_checks_below_1e_minus_6():
    g = ev.gradient_check()
    h = ev.hessian_check()
    assert g["quadratic_grad_max_rel_err"] < 1e-6
    assert g["rosenbrock_grad_max_rel_err"] < 1e-6
    assert h["quadratic_hess_max_abs_err"] < 1e-6
    assert h["rosenbrock_hess_max_rel_err"] < 1e-6
