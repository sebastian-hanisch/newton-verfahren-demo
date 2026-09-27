"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Balken-x-Achsen
nutzen echte Zahlenwerte, keine kategorischen Strings (siehe
feedback_plotly_bar_numeric_labels_vline_mismatch - Lehre aus gradient-descent-demo)."""
import numpy as np
import plotly.graph_objects as go

COLOR_PATH = "#1f77b4"
COLOR_START = "#d62728"
COLOR_OPT = "#2ca02c"
COLOR_GD = "#d62728"
COLOR_NEWTON = "#1f77b4"


def build_trajectory_figure_2d(f, trajectory, x_star, x_range, y_range, title=""):
    xs = np.linspace(x_range[0], x_range[1], 120)
    ys = np.linspace(y_range[0], y_range[1], 120)
    Z = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            Z[i, j] = f(np.array([xv, yv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=np.log1p(np.maximum(Z, 0)), showscale=False, colorscale="Blues",
        contours=dict(coloring="fill"), opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=trajectory[:, 0], y=trajectory[:, 1], mode="lines+markers", name="Pfad",
        line=dict(color=COLOR_PATH, width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=[trajectory[0, 0]], y=[trajectory[0, 1]], mode="markers", name="Start",
        marker=dict(color=COLOR_START, size=12, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], mode="markers", name="Optimum",
        marker=dict(color=COLOR_OPT, size=13, symbol="star"),
    ))
    fig.update_layout(
        title=title, xaxis=dict(range=list(x_range), fixedrange=True, title="x₁"),
        yaxis=dict(range=list(y_range), fixedrange=True, title="x₂"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_1d_landscape_figure(f, trajectory, x_star, x_range=(-2.2, 2.2), title=""):
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = [f(np.array([xv])) for xv in xs]
    traj_x = trajectory[:, 0]
    traj_y = [f(np.array([xv])) for xv in traj_x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                             line=dict(color="#8A96A6", width=2)))
    fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode="lines+markers", name="Iterierte",
                             line=dict(color=COLOR_PATH, width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[traj_x[0]], y=[traj_y[0]], mode="markers", name="Start",
                             marker=dict(color=COLOR_START, size=12, symbol="x")))
    fig.add_trace(go.Scatter(x=[x_star[0]], y=[float(f(x_star))], mode="markers", name="Optimum",
                             marker=dict(color=COLOR_OPT, size=13, symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(title="x", fixedrange=True),
        yaxis=dict(title="f(x)", fixedrange=True),
        showlegend=True, height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_convergence_figure(fvals, f_star=0.0, title="Konvergenz: f(xₖ) - f*"):
    gap = np.maximum(np.array(fvals) - f_star, 1e-300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(gap))), y=gap, mode="lines+markers", name="f(xₖ) - f*",
        line=dict(color=COLOR_PATH, width=2),
    ))
    fig.update_layout(
        title=title, xaxis=dict(title="Iteration k", fixedrange=True),
        yaxis=dict(title="f(xₖ) - f*", fixedrange=True, type="log"),
        showlegend=False, height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_newton_vs_gd_figure(newton_result, gd_result, f_star=0.0,
                              title="Newton vs. Gradientenabstieg (dieselbe Quadratik)"):
    fig = go.Figure()
    for result, name, color in ((newton_result, "Newton", COLOR_NEWTON),
                                (gd_result, "Gradientenabstieg", COLOR_GD)):
        gap = np.maximum(np.array(result.fvals) - f_star, 1e-300)
        fig.add_trace(go.Scatter(x=list(range(len(gap))), y=gap, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Iteration k", fixedrange=True),
        yaxis=dict(title="f(xₖ) - f*", fixedrange=True, type="log"),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_divergence_figure(threshold_data, title="Log-Beule: Konvergenz vs. Divergenz"):
    threshold = threshold_data["theoretical_threshold"]
    rows = threshold_data["rows"]
    x0s = [r["x0"] for r in rows]
    n_iters = [r["n_iter"] for r in rows]
    colors = ["#2ca02c" if not r["diverged"] else "#d62728" for r in rows]
    bar_width = (max(x0s) - min(x0s)) / (2.5 * len(x0s))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x0s, y=n_iters, marker=dict(color=colors), width=bar_width,
                         name="Iterationen bis Abbruch"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#5B6B80",
                  annotation_text="Schwelle 1/√3", annotation_position="top")
    fig.update_layout(
        title=title, xaxis=dict(title="Startwert x₀", fixedrange=True),
        yaxis=dict(title="Iterationen bis Konvergenz/Abbruch", fixedrange=True),
        showlegend=False, height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
