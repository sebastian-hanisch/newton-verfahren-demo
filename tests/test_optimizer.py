import numpy as np

import nm_functions as fn
import nm_optimizer as opt


def test_newton_converges_in_exactly_one_step_on_quadratic():
    A = fn.random_spd_matrix(5, 50.0, seed=2)
    f, grad, hess = fn.quadratic(A)
    x0 = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    result = opt.newton_method(f, grad, hess, x0, max_iter=10)
    assert result.n_iter == 1
    np.testing.assert_allclose(result.trajectory[-1], np.zeros(5), atol=1e-8)


def test_newton_converges_quadratically_near_rosenbrock_optimum():
    f, grad, hess = fn.rosenbrock()
    result = opt.newton_method(f, grad, hess, np.array([0.9, 0.8]), max_iter=15)
    assert result.converged
    np.testing.assert_allclose(result.trajectory[-1], np.array([1.0, 1.0]), atol=1e-6)


def test_newton_diverges_on_log_bump_above_threshold():
    f, grad, hess = fn.log_bump()
    result = opt.newton_method(f, grad, hess, np.array([0.6]), max_iter=60)
    assert result.diverged


def test_newton_converges_on_log_bump_below_threshold():
    f, grad, hess = fn.log_bump()
    result = opt.newton_method(f, grad, hess, np.array([0.5]), max_iter=60)
    assert result.converged
    assert not result.diverged
    np.testing.assert_allclose(result.trajectory[-1], np.zeros(1), atol=1e-6)


def test_trajectory_and_fvals_stay_same_length_even_on_divergence():
    f, grad, hess = fn.log_bump()
    result = opt.newton_method(f, grad, hess, np.array([0.6]), max_iter=60)
    assert len(result.trajectory) == len(result.fvals)


def test_gradient_descent_copy_converges_on_quadratic():
    A = fn.random_spd_matrix(4, 10.0, seed=0)
    f, grad, hess = fn.quadratic(A)
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    result = opt.gradient_descent(f, grad, x0, eta=0.05, max_iter=500)
    assert result.converged
