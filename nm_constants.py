"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0

FUNC_QUADRATIC = "quadratik"
FUNC_ROSENBROCK = "rosenbrock"
FUNC_LOG_BUMP = "log_beule"
FUNCTIONS = (FUNC_QUADRATIC, FUNC_ROSENBROCK, FUNC_LOG_BUMP)

DIM = 2  # feste Dimension fuer die interaktive Quadratik (2D-Kontur/Trajektorie sichtbar)

KAPPA_MIN, KAPPA_MAX, KAPPA_DEFAULT = 1.5, 500.0, 20.0
MAX_ITER_MIN, MAX_ITER_MAX, MAX_ITER_DEFAULT = 1, 60, 10

LOG_BUMP_X0_MIN, LOG_BUMP_X0_MAX, LOG_BUMP_X0_DEFAULT = -2.0, 2.0, 0.5
LOG_BUMP_THRESHOLD = 3.0 ** -0.5  # 1/sqrt(3), von Hand hergeleitet (siehe README)

ROSENBROCK_X0_MIN, ROSENBROCK_X0_MAX, ROSENBROCK_X0_DEFAULT = -1.5, 1.8, 0.9

PRESETS = {
    "quadratik_ein_schritt": dict(
        label="Quadratik — ein Schritt",
        func=FUNC_QUADRATIC, x0=0.0, kappa=200.0, max_iter=5, seed=0,
        help="Newton löst die Quadratik unabhängig von der Konditionszahl in genau einem Schritt.",
    ),
    "rosenbrock_nah": dict(
        label="Rosenbrock — nah am Optimum (quadratisch)",
        func=FUNC_ROSENBROCK, x0=0.9, kappa=KAPPA_DEFAULT, max_iter=8, seed=0,
        help="Aus der Nähe gestartet verdoppelt sich die Zahl der korrekten Nachkommastellen "
             "etwa mit jedem Schritt.",
    ),
    "log_beule_konvergiert": dict(
        label="Log-Beule — konvergiert",
        func=FUNC_LOG_BUMP, x0=0.5, kappa=KAPPA_DEFAULT, max_iter=15, seed=0,
        help="Knapp unter der Schwelle |x₀|=1/√3 konvergiert Newton zum Minimum.",
    ),
    "log_beule_divergiert": dict(
        label="Log-Beule — divergiert",
        func=FUNC_LOG_BUMP, x0=0.6, kappa=KAPPA_DEFAULT, max_iter=25, seed=0,
        help="Knapp über der Schwelle |x₀|=1/√3 läuft Newton ohne Sicherung ins Unendliche.",
    ),
}
