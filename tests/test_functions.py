import numpy as np

import nm_functions as fn


def test_random_spd_matrix_has_prescribed_condition_number():
    A = fn.random_spd_matrix(8, 50.0, seed=1)
    eigs = np.linalg.eigvalsh(A)
    kappa = eigs.max() / eigs.min()
    assert abs(kappa - 50.0) < 1e-6


def test_quadratic_minimum_is_zero_at_origin():
    A = fn.random_spd_matrix(4, 10.0, seed=0)
    f, grad, hess = fn.quadratic(A)
    assert f(np.zeros(4)) == 0.0
    np.testing.assert_allclose(grad(np.zeros(4)), np.zeros(4))
    np.testing.assert_allclose(hess(np.zeros(4)), A)


def test_rosenbrock_minimum_is_zero_at_a_a_squared():
    f, grad, hess = fn.rosenbrock(a=1.0, b=100.0)
    x_star = np.array([1.0, 1.0])
    assert f(x_star) == 0.0
    np.testing.assert_allclose(grad(x_star), np.zeros(2), atol=1e-10)


def test_log_bump_minimum_is_zero_at_origin():
    f, grad, hess = fn.log_bump()
    assert f(np.array([0.0])) == 0.0
    np.testing.assert_allclose(grad(np.array([0.0])), np.zeros(1), atol=1e-12)


def test_log_bump_gradient_matches_finite_differences():
    f, grad, hess = fn.log_bump()
    x = np.array([0.8])
    eps = 1e-6
    numeric = (f(x + np.array([eps])) - f(x - np.array([eps]))) / (2 * eps)
    np.testing.assert_allclose(grad(x)[0], numeric, rtol=1e-4)


def test_rosenbrock_hessian_matches_finite_differences_of_gradient():
    f, grad, hess = fn.rosenbrock()
    x = np.array([0.4, 0.3])
    eps = 1e-5
    H_numeric = np.zeros((2, 2))
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        H_numeric[:, i] = (grad(xp) - grad(xm)) / (2 * eps)
    np.testing.assert_allclose(hess(x), H_numeric, rtol=1e-3)
