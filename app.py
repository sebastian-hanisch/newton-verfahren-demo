"""Newton-Verfahren — Krümmung nutzen

Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton (BFGS/L-BFGS) -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 1 nutzte nur den Gradienten. Newton nutzt zusätzlich die Krümmung (Hesse-Matrix) und
konvergiert lokal quadratisch statt nur linear - unabhängig von der Konditionszahl. Der Preis:
ohne Sicherung kann der volle Schritt fern vom Optimum divergieren - hier exakt gemessen statt
nur behauptet.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import nm_constants as C
import nm_evaluation as ev
import nm_functions as fn
import nm_presets as pr
import nm_visualization as viz

st.set_page_config(page_title="Newton-Verfahren", layout="wide")

FUNC_LABELS = {C.FUNC_QUADRATIC: "Quadratik (κ regelbar)", C.FUNC_ROSENBROCK: "Rosenbrock-Funktion",
               C.FUNC_LOG_BUMP: "Log-Beule ln(1+x²)"}


@st.cache_data(show_spinner=False)
def _run(func, x0, kappa, max_iter, seed):
    import nm_optimizer as opt
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 1)
        x_start = rng.normal(size=C.DIM)
        x_star = np.zeros(C.DIM)
    elif func == C.FUNC_ROSENBROCK:
        f, grad, hess = fn.rosenbrock()
        x_start = np.array([x0, x0])
        x_star = np.array([1.0, 1.0])
    else:
        f, grad, hess = fn.log_bump()
        x_start = np.array([x0])
        x_star = np.zeros(1)
    result = opt.newton_method(f, grad, hess, x_start, max_iter=max_iter)
    return {"trajectory": result.trajectory, "fvals": result.fvals, "converged": result.converged,
            "diverged": result.diverged, "n_iter": result.n_iter, "x_star": x_star}


def _rebuild_function(func, kappa, seed):
    if func == C.FUNC_QUADRATIC:
        A = fn.random_spd_matrix(C.DIM, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        return f
    if func == C.FUNC_ROSENBROCK:
        f, grad, hess = fn.rosenbrock()
        return f
    f, grad, hess = fn.log_bump()
    return f


@st.cache_data(show_spinner=False)
def _exact_one_step_check():
    return ev.exact_one_step_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    return ev.reduction_to_gradient_descent_check()


@st.cache_data(show_spinner=False)
def _quadratic_convergence_exponents():
    return ev.quadratic_convergence_exponents()


@st.cache_data(show_spinner=False)
def _rosenbrock_wide_sweep():
    return ev.rosenbrock_wide_start_sweep()


@st.cache_data(show_spinner=False)
def _divergence_threshold():
    return ev.log_bump_divergence_threshold()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _hessian_check():
    return ev.hessian_check()


@st.cache_data(show_spinner=False)
def _newton_vs_gd(kappa, seed):
    import nm_optimizer as opt
    A = fn.random_spd_matrix(C.DIM, kappa, seed)
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=C.DIM)
    r_newton = opt.newton_method(f, grad, hess, x0, max_iter=5)
    r_gd = opt.gradient_descent(f, grad, x0, eta=0.5 / kappa, max_iter=200)
    return {"newton_fvals": r_newton.fvals, "gd_fvals": r_gd.fvals}


st.title("🏔️ Newton-Verfahren — Krümmung nutzen")
st.markdown(
    "Gradientenabstieg (Stück 1) nutzt nur die Richtung des steilsten Abstiegs. Das "
    "**Newton-Verfahren** nutzt zusätzlich die **Krümmung** (Hesse-Matrix) und springt direkt "
    "zum Minimum des lokalen quadratischen Modells: $x_{k+1}=x_k-H(x_k)^{-1}\\nabla f(x_k)$. "
    "Nahe am Optimum konvergiert das **quadratisch** statt nur linear — aber ohne Sicherung kann "
    "der volle Schritt fern vom Optimum auch danebengehen. Beides wird hier gemessen."
)
st.caption(
    "Stück 2 der 'Nichtlineare Optimierung'-Reihe. Geplante Folgestücke (noch nicht gebaut): "
    "Quasi-Newton (BFGS/L-BFGS), Lagrange/KKT, Straf-/Barriere-Verfahren, SQP, "
    "Innere-Punkte-Verfahren, Stochastische Gradientenverfahren."
)

with st.expander("So funktioniert das Newton-Verfahren", expanded=True):
    st.markdown(
        "1. Baue am aktuellen Punkt $x_k$ ein quadratisches Modell aus Gradient UND "
        "Hesse-Matrix.\n"
        "2. Springe direkt zum Minimum dieses Modells: "
        "$x_{k+1}=x_k-H(x_k)^{-1}\\nabla f(x_k)$.\n"
        "3. Ist $f$ selbst eine Quadratik, ist das Modell exakt — ein einziger Schritt genügt.\n"
        "4. Ist $f$ nur lokal quadratisch (wie die meisten glatten Funktionen), konvergiert das "
        "Verfahren nahe am Optimum sehr schnell (quadratisch), aber es gibt **keine Garantie**, "
        "dass der Schritt überhaupt in die richtige Richtung geht, wenn man weit weg startet."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    func = st.radio("Funktion", C.FUNCTIONS, format_func=lambda f: FUNC_LABELS[f],
                    key="widget_func", index=C.FUNCTIONS.index(ss["func"]),
                    on_change=pr.store_from_widget, args=("func",))
    ss["func"] = func

    if func == C.FUNC_QUADRATIC:
        kappa = st.slider("Konditionszahl κ", C.KAPPA_MIN, C.KAPPA_MAX, ss["kappa"], step=0.5,
                          key="widget_kappa", on_change=pr.store_from_widget, args=("kappa",))
        ss["kappa"] = kappa
        x0 = ss["x0"]
    elif func == C.FUNC_ROSENBROCK:
        kappa = ss["kappa"]
        x0_val = min(max(ss["x0"], C.ROSENBROCK_X0_MIN), C.ROSENBROCK_X0_MAX)
        x0 = st.slider("Startpunkt (x₀, x₀) auf der Diagonale", C.ROSENBROCK_X0_MIN,
                       C.ROSENBROCK_X0_MAX, x0_val, step=0.05, key="widget_x0",
                       on_change=pr.store_from_widget, args=("x0",))
        ss["x0"] = x0
    else:
        kappa = ss["kappa"]
        x0_val = min(max(ss["x0"], C.LOG_BUMP_X0_MIN), C.LOG_BUMP_X0_MAX)
        x0 = st.slider("Startwert x₀", C.LOG_BUMP_X0_MIN, C.LOG_BUMP_X0_MAX, x0_val, step=0.01,
                       key="widget_x0", on_change=pr.store_from_widget, args=("x0",),
                       help=f"Schwelle bei |x₀|=1/√3≈{C.LOG_BUMP_THRESHOLD:.4f}.")
        ss["x0"] = x0

    max_iter = st.slider("Max. Iterationen", C.MAX_ITER_MIN, C.MAX_ITER_MAX, ss["max_iter"],
                         key="widget_max_iter", on_change=pr.store_from_widget, args=("max_iter",))
    ss["max_iter"] = max_iter
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(func=func, x0=x0, kappa=kappa, max_iter=max_iter, seed=seed))

out = _run(func, float(x0), kappa, int(max_iter), int(seed))
f = _rebuild_function(func, kappa, int(seed))

st.markdown("---")
st.subheader("🎯 Der Weg zum Minimum")
col_left, col_right = st.columns([3, 2])
with col_left:
    if func == C.FUNC_LOG_BUMP:
        fig_traj = viz.build_1d_landscape_figure(f, out["trajectory"], out["x_star"],
                                                  title=f"Landschaft ({FUNC_LABELS[func]})")
    else:
        if func == C.FUNC_QUADRATIC:
            pad = max(1.0, float(np.abs(out["trajectory"]).max()) * 1.2)
            x_range = y_range = (-pad, pad)
        else:
            x_range, y_range = (-2.0, 2.0), (-1.0, 3.0)
        fig_traj = viz.build_trajectory_figure_2d(f, out["trajectory"], out["x_star"], x_range,
                                                  y_range, title=f"Pfad ({FUNC_LABELS[func]})")
    st.plotly_chart(fig_traj, key=f"traj_{func}_{x0}_{kappa}_{max_iter}_{seed}",
                    use_container_width=True)
with col_right:
    fig_conv = viz.build_convergence_figure(out["fvals"], 0.0)
    st.plotly_chart(fig_conv, key=f"conv_{func}_{x0}_{kappa}_{max_iter}_{seed}",
                    use_container_width=True)

st.subheader("🎯 Was am Ende steht")
m1, m2, m3 = st.columns(3)
if out["diverged"]:
    m1.metric("Konvergiert?", "Nein (divergiert)")
else:
    m1.metric("Konvergiert?", "Ja" if out["converged"] else "Nein (Max. Iterationen erreicht)")
m2.metric("Iterationen", f"{out['n_iter']}")
m3.metric("f(x) am Ende", f"{out['fvals'][-1]:.2e}" if np.isfinite(out['fvals'][-1]) else "∞")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: Newton vs. Gradientenabstieg auf derselben Quadratik")
nvg = _newton_vs_gd(20.0, 0)


class _R:
    pass


r_newton, r_gd = _R(), _R()
r_newton.fvals, r_gd.fvals = nvg["newton_fvals"], nvg["gd_fvals"]
st.plotly_chart(viz.build_newton_vs_gd_figure(r_newton, r_gd), key="newton_vs_gd_chart",
                use_container_width=True)
st.caption(
    "Bei κ=20 löst Newton dieselbe Quadratik in EINEM Schritt, Gradientenabstieg braucht "
    "hunderte Schritte — Newtons Konvergenz hängt (anders als bei Gradientenabstieg) NICHT von "
    "der Konditionszahl ab."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Hesse-Matrix ist bekannt und invertierbar | Muss bei jedem Schritt neu berechnet und "
    "invertiert werden — teuer bei großem n | Quasi-Newton (nächstes Stück) |\n"
    "| Startpunkt nahe genug am Optimum | Fern davon kann der volle Schritt divergieren, wie bei "
    "der Log-Beule ab Startwert 1/√3 (siehe 📐) | Liniensuche/Trust-Region (außerhalb dieses "
    "Rahmens) |\n"
    "| Keine Nebenbedingungen | Reine unrestringierte Minimierung | Lagrange/KKT (Stück 4) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Update:** $x_{k+1}=x_k-H(x_k)^{-1}\nabla f(x_k)$.

**Ein-Schritt-Korrektheit auf der Quadratik** $f(x)=\tfrac12x^\top Ax$: $H(x)=A$ konstant, also
$x_1=x_0-A^{-1}Ax_0=0$ — unabhängig von κ und vom Startpunkt.

**Reduktion auf Gradientenabstieg:** ersetzt man $H(x_k)$ durch $(1/\eta)I$, wird der Newton-Schritt
algebraisch identisch zu $x_{k+1}=x_k-\eta\nabla f(x_k)$ (Stück 1).
"""
    )
    err = _reduction_check()
    st.metric("Reduktion auf Gradientenabstieg: max. Abweichung der Trajektorie", f"{err:.2e}")

    exps = _quadratic_convergence_exponents()
    st.markdown("**Quadratische Konvergenz nahe dem Optimum** (Rosenbrock, Fehler je Schritt):")
    for row in exps:
        errs_str = " → ".join(f"{e:.2e}" for e in row["errors"][:6])
        st.caption(f"Start {row['x0']}: {errs_str} … (letzter Exponent ≈ {row['last_exponent']:.2f})")

    st.markdown("**Ehrlicher Befund (Plan-Korrektur):** volles Newton auf Rosenbrock konvergiert "
               "von jedem getesteten Startpunkt, aber ohne garantierte monotone Abnahme:")
    wide = _rosenbrock_wide_sweep()
    w1, w2, w3 = st.columns(3)
    w1.metric("Konvergiert", f"{wide['n_converged']}/{wide['n_total']}")
    w2.metric("Nicht-monoton (f steigt zwischendurch)", f"{wide['n_non_monotone']}/{wide['n_total']}")
    w3.metric("Max. Iterationen gebraucht", f"{wide['max_iters_used']}")

    st.markdown("**Echte Divergenz: die Log-Beule** $f(x)=\\ln(1+x^2)$ — Fixpunkt-Iteration "
               "$x_{k+1}=\\dfrac{2x_k^3}{x_k^2-1}$, Schwelle exakt bei $|x_0|=1/\\sqrt3$:")
    thr = _divergence_threshold()
    st.plotly_chart(viz.build_divergence_figure(thr), key="divergence_chart",
                    use_container_width=True)

    g1, g2 = st.columns(2)
    grad_err = _gradient_check()
    hess_err = _hessian_check()
    g1.metric("Gradienten-Check (Quadratik/Rosenbrock)",
              f"{grad_err['quadratic_grad_max_rel_err']:.1e} / {grad_err['rosenbrock_grad_max_rel_err']:.1e}")
    g2.metric("Hesse-Check (Quadratik/Rosenbrock)",
              f"{hess_err['quadratic_hess_max_abs_err']:.1e} / {hess_err['rosenbrock_hess_max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). "
        "Springer."
    )
    st.caption(
        "Implementiert in `nm_functions.py` (Testfunktionen), `nm_optimizer.py` (Newton, "
        "Gradientenabstieg-Kopie), `nm_evaluation.py` (Korrektheits-Kette, Sweeps), "
        "`nm_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
