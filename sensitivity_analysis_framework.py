"""
Sensitivity Analysis Framework for Stochastic Simulators
=========================================================

Sobol + PAWN on E[Y|theta] and Var[Y|theta].
Threshold exceedance analysis with interactive threshold input:
  - PAWN on exceedance, PRIM box peeling, pairwise heatmaps
  - All recomputed on-the-fly when threshold changes (no re-simulation)
  - Time-window RangeSlider for temporal interaction analysis
  - Conditioning RangeSliders to restrict marginalisation in heatmaps

Requirements:
    pip install SALib numpy plotly pandas dash
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from SALib.analyze import pawn as pawn_analyze
from itertools import combinations
import warnings
from dash import Dash, dcc, html, Input, Output, Patch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_SERIES_AXIS  = 100   # default number of points if series_axis is given as int
N_SAMPLES      = 258
N_REPLICAS     = 100
CONFIDENCE_LVL = 0.95
SEED           = 42

PRIM_ALPHA       = 0.05
PRIM_MIN_SUPPORT = 0.05
HEATMAP_BINS     = 25
MIN_SAMPLES_HEATMAP = 30   # minimum conditioned samples to compute a heatmap

THRESHOLD_INIT   = None   # initial value in dashboard (None for empty)

MAX_DISPLAY      = 25


# =============================================================================
# ENGINE  (threshold-independent — runs once)
# =============================================================================
def _sobol_at_each_timestep(problem, Y_matrix, n_points, conf_level):
    names = problem["names"]; num_vars = problem["num_vars"]
    pairs = list(combinations(range(num_vars), 2))
    S1 = np.zeros((n_points, num_vars)); ST = np.zeros_like(S1)
    S1c = np.zeros_like(S1); STc = np.zeros_like(S1)
    S2 = {(names[i],names[j]): np.zeros(n_points) for i,j in pairs}
    S2c = {(names[i],names[j]): np.zeros(n_points) for i,j in pairs}
    for t in range(n_points):
        col = Y_matrix[:, t]
        if np.std(col) < 1e-15:
            continue
        try:
            Si = sobol.analyze(problem, col, calc_second_order=True,
                               conf_level=conf_level, print_to_console=False)
            S1[t,:] = Si["S1"]; ST[t,:] = Si["ST"]
            S1c[t,:] = Si["S1_conf"]; STc[t,:] = Si["ST_conf"]
            for idx,(i,j) in enumerate(pairs):
                S2[(names[i],names[j])][t] = Si["S2"][i,j]
                S2c[(names[i],names[j])][t] = Si["S2_conf"][i,j]
        except (ValueError, TypeError, FloatingPointError):
            pass
    ts = np.arange(n_points)
    return {"S1": pd.DataFrame(S1, columns=names, index=ts),
            "ST": pd.DataFrame(ST, columns=names, index=ts),
            "S1_conf": pd.DataFrame(S1c, columns=names, index=ts),
            "ST_conf": pd.DataFrame(STc, columns=names, index=ts),
            "S2": S2, "S2_conf": S2c}


def _pawn_at_each_timestep(problem, X, Y_matrix, n_points):
    names = problem["names"]; num_vars = problem["num_vars"]
    med = np.zeros((n_points, num_vars))
    mn = np.zeros_like(med); mx = np.zeros_like(med); cv = np.zeros_like(med)
    for t in range(n_points):
        col = Y_matrix[:, t]
        if np.std(col) < 1e-15:
            continue
        try:
            Si = pawn_analyze.analyze(problem, X, col, S=10, print_to_console=False)
            med[t,:] = Si["median"]; mn[t,:] = Si["minimum"]
            mx[t,:] = Si["maximum"]; cv[t,:] = Si["CV"]
        except Exception:
            pass
    ts = np.arange(n_points)
    return {"median": pd.DataFrame(med, columns=names, index=ts),
            "minimum": pd.DataFrame(mn, columns=names, index=ts),
            "maximum": pd.DataFrame(mx, columns=names, index=ts),
            "CV": pd.DataFrame(cv, columns=names, index=ts)}


# --------------- PRIM (Patient Rule Induction Method) -----------------------
def _prim_peel(X, y, names, alpha=PRIM_ALPHA, min_support=PRIM_MIN_SUPPORT):
    n = len(y)
    mask = np.ones(n, dtype=bool)
    bounds_lo = X.min(axis=0).copy()
    bounds_hi = X.max(axis=0).copy()
    n_total_pos = y.sum()
    num_vars = X.shape[1]

    trajectory = []
    trajectory.append({
        "bounds": list(zip(bounds_lo.tolist(), bounds_hi.tolist())),
        "density": y[mask].mean() if mask.sum() > 0 else 0,
        "coverage": y[mask].sum() / max(n_total_pos, 1),
        "support": mask.sum() / n,
        "n_in_box": int(mask.sum()),
    })

    while mask.sum() > max(int(min_support * n), 10):
        best_density = -1
        best_mask = None
        best_bounds = None
        for j in range(num_vars):
            vals = X[mask, j]
            cutoff_lo = np.percentile(vals, alpha * 100)
            new_mask_lo = mask.copy()
            new_mask_lo &= (X[:, j] >= cutoff_lo)
            if new_mask_lo.sum() > 0 and new_mask_lo.sum() < mask.sum():
                d = y[new_mask_lo].mean()
                if d > best_density:
                    best_density = d
                    best_mask = new_mask_lo
                    new_lo = bounds_lo.copy(); new_hi = bounds_hi.copy()
                    new_lo[j] = cutoff_lo
                    best_bounds = (new_lo, new_hi)
            cutoff_hi = np.percentile(vals, (1 - alpha) * 100)
            new_mask_hi = mask.copy()
            new_mask_hi &= (X[:, j] <= cutoff_hi)
            if new_mask_hi.sum() > 0 and new_mask_hi.sum() < mask.sum():
                d = y[new_mask_hi].mean()
                if d > best_density:
                    best_density = d
                    best_mask = new_mask_hi
                    new_lo = bounds_lo.copy(); new_hi = bounds_hi.copy()
                    new_hi[j] = cutoff_hi
                    best_bounds = (new_lo, new_hi)
        if best_mask is None or best_density <= trajectory[-1]["density"]:
            break
        mask = best_mask
        bounds_lo, bounds_hi = best_bounds
        trajectory.append({
            "bounds": list(zip(bounds_lo.tolist(), bounds_hi.tolist())),
            "density": best_density,
            "coverage": y[mask].sum() / max(n_total_pos, 1),
            "support": mask.sum() / n,
            "n_in_box": int(mask.sum()),
        })
    return trajectory


def _prim_for_window(param_samples, exceed_ts, names, t_start, t_end,
                     sample_mask=None, alpha=PRIM_ALPHA, min_support=PRIM_MIN_SUPPORT):
    """Run PRIM on exceedance restricted to [t_start, t_end], optionally on a subset."""
    if sample_mask is not None:
        ps = param_samples[sample_mask]
        et = exceed_ts[sample_mask]
    else:
        ps = param_samples
        et = exceed_ts
    window_exceed = et[:, t_start:t_end+1]
    labels = (window_exceed.max(axis=1) > 0.5).astype(float)
    if labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
        return None
    trajectory = _prim_peel(ps, labels, names, alpha, min_support)
    best_box = trajectory[-1]
    for box in trajectory:
        if box["density"] >= 0.8:
            best_box = box
            break
    return {"trajectory": trajectory, "best_box": best_box}


def _heatmap_single_pair(param_samples, max_exceed, idx_i, idx_j,
                         sample_mask=None, n_bins=HEATMAP_BINS):
    """Compute one heatmap for pair (i,j) with an optional sample mask."""
    if sample_mask is not None:
        ps = param_samples[sample_mask]
        me = max_exceed[sample_mask]
    else:
        ps = param_samples
        me = max_exceed
    if len(me) < MIN_SAMPLES_HEATMAP:
        return None  # too few samples
    xi = ps[:, idx_i]; xj = ps[:, idx_j]
    xi_edges = np.linspace(xi.min(), xi.max(), n_bins + 1)
    xj_edges = np.linspace(xj.min(), xj.max(), n_bins + 1)
    grid = np.full((n_bins, n_bins), np.nan)
    for bi in range(n_bins):
        for bj in range(n_bins):
            mask = ((xi >= xi_edges[bi]) & (xi < xi_edges[bi+1]) &
                    (xj >= xj_edges[bj]) & (xj < xj_edges[bj+1]))
            if bi == n_bins - 1:
                mask |= (xi >= xi_edges[bi]) & (xi <= xi_edges[bi+1]) & \
                        (xj >= xj_edges[bj]) & (xj < xj_edges[bj+1])
            if bj == n_bins - 1:
                mask |= (xi >= xi_edges[bi]) & (xi < xi_edges[bi+1]) & \
                        (xj >= xj_edges[bj]) & (xj <= xj_edges[bj+1])
            if mask.sum() > 0:
                grid[bj, bi] = me[mask].mean()
    xi_mids = 0.5*(xi_edges[:-1]+xi_edges[1:])
    xj_mids = 0.5*(xj_edges[:-1]+xj_edges[1:])
    return {"grid": grid, "x_mids": xi_mids, "y_mids": xj_mids,
            "n_samples": len(me)}


def run_sensitivity_analysis(simulator_fn, problem, series_axis: "list | range | np.ndarray" = N_SERIES_AXIS,
                              n_samples=N_SAMPLES, n_replicas=N_REPLICAS,
                              conf_level=CONFIDENCE_LVL, seed=SEED):
    """Run simulations + Sobol/PAWN on mean and variance. Threshold-independent.

    Parameters
    ----------
    simulator_fn : function (params: dict, series_axis: np.ndarray, rng: np.random.Generator) -> np.ndarray
        User-defined simulator function.  Must take a parameter dict, the series_axis array, and a random number generator, and return an array of shape (len(series_axis),) corresponding to the model evaluation at each point in the series_axis.
    problem : dict
        Problem definition containing parameter names and bounds. An example is
        problem = {"num_vars":3, "names":["P0","P1","P2"], "bounds":[[0,10],[0,5],[0,5]]}
        
    series_axis : list | range | np.ndarray
        Defines the x-axis of the output series.  The framework passes this
        array directly to ``simulator_fn``, which must return an array of
        shape ``(len(series_axis),)``.  Any dense-grid vs. sparse-axis logic
        is the simulator's responsibility.
    """
    if series_axis is None or isinstance(series_axis, (int,np.integer)):
        TypeError("series_axis must be a list, range, or np.ndarray of values, not an int.")
    else:
        series_axis = np.asarray(series_axis)
    n_points = len(series_axis)

    names = problem["names"]; num_vars = problem["num_vars"]
    print(f"Saltelli samples (N={n_samples}, D={num_vars}) ...")
    param_samples = sobol_sample.sample(problem, n_samples, calc_second_order=True)
    n_ps = param_samples.shape[0]
    print(f"  {n_ps} parameter sets")

    print(f"Simulating {n_ps} x {n_replicas} = {n_ps*n_replicas} runs ...")
    all_ts = np.empty((n_ps, n_replicas, n_points))
    base_rng = np.random.default_rng(seed)
    for i in range(n_ps):
        p = {name: param_samples[i,j] for j,name in enumerate(names)}
        for r in range(n_replicas):
            rng = np.random.default_rng(base_rng.integers(0, 2**31))
            all_ts[i, r, :] = simulator_fn(p, series_axis, rng)
        if (i+1) % max(1, n_ps//10) == 0:
            print(f"  {i+1}/{n_ps}")

    mean_ts = np.mean(all_ts, axis=1); var_ts = np.var(all_ts, axis=1)
    tv_mean = np.var(mean_ts, axis=0); tv_var = np.var(var_ts, axis=0)

    print("Sobol on E[Y|theta] ..."); sm = _sobol_at_each_timestep(problem, mean_ts, n_points, conf_level)
    print("Sobol on Var[Y|theta] ..."); sv = _sobol_at_each_timestep(problem, var_ts, n_points, conf_level)
    print("PAWN on E[Y|theta] ..."); pm = _pawn_at_each_timestep(problem, param_samples, mean_ts, n_points)
    print("PAWN on Var[Y|theta] ..."); pv = _pawn_at_each_timestep(problem, param_samples, var_ts, n_points)

    print("Done.")
    return {"param_samples": param_samples, "timeseries": all_ts,
            "mean_ts": mean_ts, "var_ts": var_ts,
            "total_var_of_mean": tv_mean, "total_var_of_var": tv_var,
            "sobol_mean": sm, "sobol_var": sv,
            "pawn_mean": pm, "pawn_var": pv,
            "problem": problem, "series_axis": series_axis,
            "n_replicas": n_replicas}


def print_summary(results):
    names = results["problem"]["names"]
    for label, sk in [("MEAN","sobol_mean"),("VAR","sobol_var")]:
        s = results[sk]; S1=s["S1"]; ST=s["ST"]
        print(f"\n{'='*60}\n  Sobol -- {label}\n{'='*60}")
        a1=S1.mean(); aT=ST.mean()
        for n in a1.sort_values(ascending=False).index:
            print(f"    {n:>8s}: S1={a1[n]:+.4f}  ST={aT[n]:+.4f}")
    for label, pk in [("MEAN","pawn_mean"),("VAR","pawn_var")]:
        avg=results[pk]["median"].mean()
        print(f"\n  PAWN ({label}) time-averaged:")
        for n in avg.sort_values(ascending=False).index:
            print(f"    {n:>8s}: {avg[n]:.4f}")
    print("="*60)


# =============================================================================
# DASH DASHBOARD
# =============================================================================
def build_dash_app(results, initial_threshold=THRESHOLD_INIT):
    prob = results["problem"]; names = prob["names"]
    num_vars = prob["num_vars"]; series_axis = results["series_axis"]
    n_ts = len(series_axis)
    param_samples = results["param_samples"]; mean_ts = results["mean_ts"]
    all_ts = results["timeseries"]
    timesteps = series_axis
    pair_keys = [(names[i],names[j]) for i in range(num_vars) for j in range(i+1,num_vars)]
    colors = ["#636EFA","#EF553B","#00CC96","#AB63FA",
              "#FFA15A","#19D3F3","#FF6692","#B6E880"]

    # Actual parameter ranges from samples (robust to non-uniform dists where
    # problem["bounds"] encodes distribution params, not min/max).
    samp_mins = param_samples.min(axis=0)
    samp_maxs = param_samples.max(axis=0)
    actual_bounds = list(zip(samp_mins.tolist(), samp_maxs.tolist()))

    def _fill(c, alpha):
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        return f"rgba({r},{g},{b},{alpha})"

    # =====================================================================
    # LEFT: Sobol + PAWN figure (static, built once)
    # =====================================================================
    def _make_sobol_fig():
        fig = make_subplots(
            rows=5, cols=2, vertical_spacing=0.06, horizontal_spacing=0.08,
            row_heights=[0.22, 0.19, 0.17, 0.22, 0.20],
            subplot_titles=(
                "Abs. Variance (stacked) \u2014 Mean E[Y|\u03b8]",
                "Abs. Variance (stacked) \u2014 Var[Y|\u03b8]",
                "First-Order S1 (stacked) \u2014 Mean E[Y|\u03b8]",
                "First-Order S1 (stacked) \u2014 Var[Y|\u03b8]",
                "Second-Order S2 \u2014 Mean E[Y|\u03b8]",
                "Second-Order S2 \u2014 Var[Y|\u03b8]",
                "Total-Order ST & S1 \u2014 Mean E[Y|\u03b8]",
                "Total-Order ST & S1 \u2014 Var[Y|\u03b8]",
                "PAWN (median, min\u2013max) \u2014 Mean E[Y|\u03b8]",
                "PAWN (median, min\u2013max) \u2014 Var[Y|\u03b8]",
            ))
        ci_indices = []
        for ci, (sk, tvk, pk) in enumerate([
            ("sobol_mean","total_var_of_mean","pawn_mean"),
            ("sobol_var","total_var_of_var","pawn_var")], start=1):
            sob = results[sk]; S1=sob["S1"]; ST=sob["ST"]
            S1c=sob["S1_conf"]; STc=sob["ST_conf"]
            S2=sob["S2"]; S2c=sob["S2_conf"]
            Vt=results[tvk]; pw=results[pk]
            sh=(ci==1); pks=list(S2.keys())
            VS1={n: S1[n].clip(lower=0).values*Vt for n in names}
            for i,n in enumerate(names):
                c=colors[i%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,y=VS1[n],mode="lines",
                    stackgroup=f"Va{ci}",name=f"V_S1({n})",showlegend=sh,
                    line=dict(width=0.5,color=c),legendgroup=f"V_{n}",
                    legendgrouptitle_text="Abs. Variance" if(sh and i==0) else None,
                ), row=1, col=ci)
            for pi,pr in enumerate(pks):
                c=colors[(num_vars+pi)%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,
                    y=np.clip(S2[pr],0,None)*Vt,mode="lines",
                    stackgroup=f"Va{ci}",name=f"V_S2({pr[0]}\u00d7{pr[1]})",
                    showlegend=sh,line=dict(width=0.5,color=c,dash="dash"),
                    legendgroup=f"VS2_{pr[0]}_{pr[1]}",
                ), row=1, col=ci)
            ho=np.clip(Vt-sum(VS1[n] for n in names)-sum(np.clip(S2[p],0,None)*Vt for p in pks),0,None)
            fig.add_trace(go.Scatter(x=timesteps,y=ho,mode="lines",
                stackgroup=f"Va{ci}",name="V(3rd+ order)",showlegend=sh,
                line=dict(width=0.5,color="rgba(180,180,180,0.8)"),
                fillcolor="rgba(180,180,180,0.3)",legendgroup="Vhi",
            ), row=1, col=ci)
            for i,n in enumerate(names):
                c=colors[i%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,y=S1[n].clip(lower=0),
                    mode="lines",stackgroup=f"s1{ci}",name=f"S1({n})",
                    showlegend=sh,line=dict(width=0.5,color=c),
                    legendgroup=f"S1_{n}",
                    legendgrouptitle_text="First-Order S1" if(sh and i==0) else None,
                ), row=2, col=ci)
            ai=(1-S1.clip(lower=0).sum(axis=1)).clip(lower=0)
            fig.add_trace(go.Scatter(x=timesteps,y=ai,mode="lines",
                stackgroup=f"s1{ci}",name="Interactions (1\u2212\u03a3S1)",showlegend=sh,
                line=dict(width=0.5,color="rgba(180,180,180,0.8)"),
                fillcolor="rgba(180,180,180,0.3)",legendgroup="s1int",
            ), row=2, col=ci)
            for pi,pr in enumerate(pks):
                c=colors[(num_vars+pi)%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,y=S2[pr],mode="lines",
                    name=f"S2({pr[0]}\u00d7{pr[1]})",showlegend=sh,
                    line=dict(width=2,color=c,dash="dash"),
                    legendgroup=f"S2_{pr[0]}_{pr[1]}",
                    legendgrouptitle_text="Second-Order S2" if(sh and pi==0) else None,
                ), row=3, col=ci)
                ci_idx = len(fig.data)
                u=S2[pr]+S2c[pr]; l=np.clip(S2[pr]-S2c[pr],0,None)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([timesteps,timesteps[::-1]]),
                    y=np.concatenate([u,l[::-1]]),fill="toself",
                    fillcolor=_fill(c,0.1),line=dict(width=0),showlegend=False,
                    legendgroup=f"S2_{pr[0]}_{pr[1]}",hoverinfo="skip",opacity=0.5,
                ), row=3, col=ci)
                ci_indices.append(ci_idx)
            for i,n in enumerate(names):
                c=colors[i%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,y=ST[n].values,mode="lines",
                    name=f"ST({n})",showlegend=sh,line=dict(width=3,color=c),
                    legendgroup=f"ST_{n}",
                    legendgrouptitle_text="Total-Order ST" if(sh and i==0) else None,
                ), row=4, col=ci)
                ci_idx = len(fig.data)
                u=(ST[n]+STc[n]).values; l=(ST[n]-STc[n]).clip(lower=0).values
                fig.add_trace(go.Scatter(
                    x=np.concatenate([timesteps,timesteps[::-1]]),
                    y=np.concatenate([u,l[::-1]]),fill="toself",
                    fillcolor=_fill(c,0.12),line=dict(width=0),showlegend=False,
                    legendgroup=f"ST_{n}",hoverinfo="skip",opacity=0.5,
                ), row=4, col=ci)
                ci_indices.append(ci_idx)
                fig.add_trace(go.Scatter(x=timesteps,y=S1[n].clip(lower=0).values,
                    mode="lines",showlegend=False,name=f"S1({n})",
                    line=dict(width=1,color=c,dash="dot"),
                    legendgroup=f"ST_{n}",opacity=0.5,
                ), row=4, col=ci)
            pmed=pw["median"]; pmn=pw["minimum"]; pmx=pw["maximum"]
            for i,n in enumerate(names):
                c=colors[i%len(colors)]
                fig.add_trace(go.Scatter(x=timesteps,y=pmed[n].values,
                    mode="lines",name=f"PAWN({n})",showlegend=sh,
                    line=dict(width=2.5,color=c),legendgroup=f"PW_{n}",
                    legendgrouptitle_text="PAWN median" if(sh and i==0) else None,
                ), row=5, col=ci)
                ci_idx = len(fig.data)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([timesteps,timesteps[::-1]]),
                    y=np.concatenate([pmx[n].values,pmn[n].values[::-1]]),
                    fill="toself",fillcolor=_fill(c,0.12),
                    line=dict(width=0),showlegend=False,
                    legendgroup=f"PW_{n}",hoverinfo="skip",opacity=0.5,
                ), row=5, col=ci)
                ci_indices.append(ci_idx)
        n_tr = len(fig.data)
        vis_all = [True]*n_tr; vis_no_ci = [True]*n_tr
        for idx in ci_indices: vis_no_ci[idx] = False
        fig.update_layout(height=1800, template="plotly_white",
            legend=dict(orientation="v",y=0.5,yanchor="middle",x=1.02,
                        xanchor="left",font=dict(size=10),groupclick="togglegroup"),
            margin=dict(t=40,b=50,l=60,r=180),hovermode="x unified",
            updatemenus=[dict(type="buttons",direction="left",
                x=0.0,xanchor="left",y=1.02,yanchor="bottom",
                buttons=[
                    dict(label="\u2611 CI bands",method="update",args=[{"visible":vis_all}]),
                    dict(label="\u2610 CI bands",method="update",args=[{"visible":vis_no_ci}]),
                ], font=dict(size=11),bgcolor="white",bordercolor="#ccc")])
        for a in fig.layout.annotations: a.font.size=12
        fig.update_yaxes(title_text="Var (units\u00b2)",row=1,col=1)
        fig.update_yaxes(title_text="S1",row=2,col=1,range=[0,1.05])
        fig.update_yaxes(range=[0,1.05],row=2,col=2)
        fig.update_yaxes(title_text="S2",row=3,col=1)
        fig.update_yaxes(title_text="ST",row=4,col=1)
        fig.update_yaxes(title_text="PAWN",row=5,col=1,range=[0,1.05])
        fig.update_yaxes(range=[0,1.05],row=5,col=2)
        fig.update_xaxes(title_text="Series",row=5,col=1)
        fig.update_xaxes(title_text="Series",row=5,col=2)
        return fig

    sobol_fig = _make_sobol_fig()

    # =====================================================================
    # Pre-compute helpers
    # =====================================================================
    flat = all_ts.reshape(-1, n_ts)
    ylo = float(np.percentile(flat,0.5)); yhi = float(np.percentile(flat,99.5))
    ypad = (yhi-ylo)*0.05; y_range = [ylo-ypad, yhi+ypad]
    pmins = param_samples.min(0); pmaxs = param_samples.max(0)
    pranges = pmaxs-pmins; pranges[pranges==0]=1.0
    normed = (param_samples-pmins)/pranges

    print("Pre-computing quantiles for fluid UI interactions...")
    p05_all = np.percentile(all_ts, 5, axis=1)
    p25_all = np.percentile(all_ts, 25, axis=1)
    p75_all = np.percentile(all_ts, 75, axis=1)
    p95_all = np.percentile(all_ts, 95, axis=1)
    med_all = np.median(all_ts, axis=1)
    global_mean = np.mean(all_ts, axis=(0,1))
    global_q05 = np.quantile(all_ts, 0.05, axis=(0,1))
    global_q95 = np.quantile(all_ts, 0.95, axis=(0,1))

    # =====================================================================
    # Build initial sim figure (Patch-updated)
    # =====================================================================
    stride = max(1, n_ts // 200)
    ts_d = timesteps[::stride]
    n_spaghetti = min(MAX_DISPLAY, all_ts.shape[1])
    IDX_BAND95 = 0; IDX_BAND75 = 1; IDX_MEDIAN = 2
    IDX_SPAG_START = 3; IDX_MU = IDX_SPAG_START + n_spaghetti
    IDX_GLOBAL_MEAN = IDX_MU + 1; IDX_GLOBAL_BAND = IDX_GLOBAL_MEAN + 1
    IDX_THRESHOLD = IDX_GLOBAL_BAND + 1

    def _build_initial_sim_fig():
        mid_vals = np.array([(lo+hi)/2.0 for lo,hi in actual_bounds])
        mid_n = (mid_vals - pmins) / pranges
        bi = int(np.argmin(np.linalg.norm(normed - mid_n, axis=1)))
        mp = param_samples[bi]
        lab = ", ".join(f"{names[j]}={mp[j]:.3g}" for j in range(num_vars))
        reps = all_ts[bi]; fig = go.Figure()
        band_x = np.concatenate([ts_d, ts_d[::-1]])
        p05 = p05_all[bi][::stride]; p95 = p95_all[bi][::stride]
        fig.add_trace(go.Scatter(x=band_x, y=np.concatenate([p95, p05[::-1]]),
            fill="toself", fillcolor="rgba(0,100,250,0.10)",
            line=dict(width=0), name="Replica 5\u201395%", hoverinfo="skip"))
        p25 = p25_all[bi][::stride]; p75 = p75_all[bi][::stride]
        fig.add_trace(go.Scatter(x=band_x, y=np.concatenate([p75, p25[::-1]]),
            fill="toself", fillcolor="rgba(0,100,250,0.20)",
            line=dict(width=0), name="Replica 25\u201375%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ts_d, y=med_all[bi][::stride], mode="lines",
            line=dict(color="rgba(0,100,250,0.6)", width=1.5, dash="dot"),
            name="Replica median"))
        for ri in range(n_spaghetti):
            fig.add_trace(go.Scattergl(x=ts_d, y=reps[ri][::stride], mode="lines",
                line=dict(color="rgba(0,100,250,0.08)", width=0.8),
                showlegend=(ri==0), name="Replicas" if ri==0 else None, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ts_d, y=mean_ts[bi][::stride], mode="lines",
            line=dict(color="blue", width=3), name="E[Y|\u03b8]"))
        fig.add_trace(go.Scatter(x=ts_d, y=global_mean[::stride], mode="lines",
            line=dict(color="black", width=3), name="E[Y]"))
        fig.add_trace(go.Scatter(x=band_x,
            y=np.concatenate([global_q95[::stride], global_q05[::stride][::-1]]),
            fill="toself", fillcolor="rgba(120,120,120,0.35)", line=dict(width=0),
            name="Global 5\u201395%", hoverinfo="skip"))
        thr_y = initial_threshold if initial_threshold is not None else 0
        thr_vis = initial_threshold is not None
        fig.add_trace(go.Scatter(x=[float(ts_d[0]), float(ts_d[-1])], y=[thr_y, thr_y],
            mode="lines", line=dict(color="#B22222", width=2, dash="dash"),
            name="Threshold", visible=thr_vis, showlegend=thr_vis))
        fig.update_layout(title=f"Simulation \u2014 nearest: {lab}",
            yaxis=dict(range=y_range, title="Output"),
            xaxis=dict(title="Series"), template="plotly_white", height=400,
            margin=dict(t=50, b=40, l=60, r=30),
            legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5),
            uirevision="constant")
        return fig

    initial_sim_fig = _build_initial_sim_fig()

    # =====================================================================
    # Dash layout
    # =====================================================================
    app = Dash(__name__)

    sliders = []
    for i,n in enumerate(names):
        lo,hi = actual_bounds[i]; mid=(lo+hi)/2; step=(hi-lo)/100
        sliders.append(html.Div([
            html.Label(n, style={"width":"80px","fontWeight":"bold",
                                 "fontSize":"14px","display":"inline-block"}),
            dcc.Slider(id=f"slider-{n}",min=lo,max=hi,step=step,value=mid,
                       tooltip={"always_visible":True,"placement":"bottom"},
                       marks={lo:f"{lo:.2g}",hi:f"{hi:.2g}"}),
        ], style={"marginBottom":"8px"}))

    heatmap_graphs = [
        dcc.Graph(id=f"heatmap-{pidx}",
                  style={"marginBottom":"5px", "minHeight":"380px"})
        for pidx in range(len(pair_keys))]

    def _fmt_axis(v):
        return str(int(v)) if float(v) == int(v) else f"{v:.4g}"

    # Time-window slider uses integer indices (0..n_ts-1) for exact snapping;
    # mark *labels* show the corresponding series_axis values.
    mark_step = max(1, n_ts // 5)
    range_marks = {i: _fmt_axis(series_axis[i]) for i in range(0, n_ts, mark_step)}
    range_marks[n_ts - 1] = _fmt_axis(series_axis[-1])

    # --- Conditioning RangeSliders ---
    cond_sliders = []
    for i, n in enumerate(names):
        slo = float(samp_mins[i]); shi = float(samp_maxs[i])
        step = (shi - slo) / 100
        cond_sliders.append(html.Div([
            html.Label(n, style={"width":"50px","fontWeight":"bold","fontSize":"12px",
                                 "display":"inline-block","color":colors[i%len(colors)]}),
            dcc.RangeSlider(
                id=f"cond-slider-{n}",
                min=slo, max=shi, step=step,
                value=[slo, shi],
                marks={slo: f"{slo:.2g}", shi: f"{shi:.2g}"},
                tooltip={"always_visible": False, "placement": "bottom"},
                allowCross=False,
            ),
        ], style={"marginBottom":"4px"}))

    right_children = [
        html.H4("Threshold Exceedance Analysis",
                 style={"textAlign":"center","color":"#B22222","marginBottom":"6px"}),
        html.Div([
            html.Label("Threshold c = ",
                       style={"fontWeight":"bold","fontSize":"14px","color":"#B22222",
                              "marginRight":"6px"}),
            dcc.Input(id="threshold-input", type="number",
                      value=initial_threshold, placeholder="Enter threshold...",
                      debounce=True,
                      style={"width":"100px","fontSize":"14px","padding":"4px 8px",
                             "border":"1px solid #dcc","borderRadius":"4px"}),
            html.Span(" (press Enter)",
                       style={"fontSize":"11px","color":"#888","marginLeft":"8px"}),
        ], style={"textAlign":"center","marginBottom":"10px"}),

        html.Div([
            html.Label("PRIM / Heatmap time window:",
                       style={"fontWeight":"bold","fontSize":"13px","color":"#555"}),
            dcc.RangeSlider(id="time-window-slider",
                min=0, max=n_ts-1, step=1,
                value=[0, n_ts-1],
                marks=range_marks,
                tooltip={"always_visible":True,"placement":"bottom"},
                allowCross=False),
        ], style={"padding":"8px 5px","margin":"5px 0 10px 0",
                  "backgroundColor":"#fef8f8","borderRadius":"6px",
                  "border":"1px solid #dcc"}),

        dcc.Loading(id="loading-threshold", type="circle", color="#B22222", children=[
            dcc.Graph(id="threshold-plot"),

            html.Div(id="prim-text"),

            html.H5("Pairwise Interaction Heatmaps",
                     style={"textAlign":"center","color":"#555","marginTop":"12px"}),
            html.P("Dashed box = PRIM danger zone projected onto each pair.",
                   style={"textAlign":"center","fontSize":"11px","color":"#888"}),

            # Conditioning sliders (collapsible)
            html.Details([
                html.Summary("Condition on parameter ranges (click to expand)",
                             style={"fontWeight":"bold","fontSize":"12px",
                                    "cursor":"pointer","color":"#555"}),
                html.P("Restrict which samples are included when marginalising. "
                       "For pair (A, B), only variables other than A and B are filtered. "
                       "PRIM uses all constraints.",
                       style={"fontSize":"10px","color":"#888","margin":"4px 0 6px 0"}),
                html.Div(cond_sliders),
                html.Div(id="cond-status",
                         style={"fontSize":"11px","color":"#555","marginTop":"4px",
                                "textAlign":"center"}),
            ], style={"margin":"6px 5px 10px 5px","padding":"6px",
                      "backgroundColor":"#f5f0f0","borderRadius":"6px",
                      "border":"1px solid #dcc"}, open=False),
        ] + heatmap_graphs),
    ]

    panels = html.Div([
        html.Div([
            html.H4("Sobol & PAWN Sensitivity Indices",
                     style={"textAlign":"center","color":"#333","marginBottom":"2px"}),
            html.P("Left: Mean E[Y|\u03b8].  Right: Variance Var[Y|\u03b8].",
                   style={"textAlign":"center","fontSize":"12px","color":"#600"}),
            dcc.Graph(id="sobol-plot",figure=sobol_fig),
        ], style={"flex":"5","minWidth":"0"}),
        html.Div(style={"width":"5px","backgroundColor":"#999",
                        "margin":"0 10px","borderRadius":"3px","flexShrink":"0"}),
        html.Div(right_children, style={"flex":"3","minWidth":"0"}),
    ], style={"display":"flex","flexDirection":"row","alignItems":"stretch"})

    app.layout = html.Div([
        html.H2("Stochastic Simulator \u2014 Global Sensitivity Analysis",
                 style={"textAlign":"center","color":"#333","marginBottom":"5px"}),
        html.Hr(),
        html.Div([
            html.H4("Parameter Space Explorer",style={"marginBottom":"10px","color":"#555"}),
            html.P("Drag sliders to explore. Shows replicas from nearest sampled combination.",
                   style={"fontSize":"12px","color":"#888","marginBottom":"10px"}),
            html.Div(sliders, style={"maxWidth":"700px"}),
        ], style={"padding":"15px 30px","backgroundColor":"#f9f9f9",
                  "borderRadius":"8px","margin":"10px 20px","border":"1px solid #e0e0e0"}),
        dcc.Graph(id="sim-plot", figure=initial_sim_fig, style={"marginTop":"10px"}),
        html.Hr(),
        panels,
    ], style={"fontFamily":"Arial, sans-serif","margin":"0 auto","maxWidth":"2800px"})

    # =====================================================================
    # Cache
    # =====================================================================
    _cache = {"threshold": None, "exceed_ts": None, "pawn_exceed": None}

    # =====================================================================
    # Callbacks
    # =====================================================================

    # --- Sim plot (Patch) ---
    @app.callback(Output("sim-plot","figure"),
                  [Input(f"slider-{n}","value") for n in names] +
                  [Input("threshold-input","value")])
    def update_sim(*args):
        vals = args[:num_vars]; thr_value = args[num_vars]
        tgt = np.array(vals)
        tgt_n = (tgt - pmins) / pranges
        bi = int(np.argmin(np.linalg.norm(normed - tgt_n, axis=1)))
        mp = param_samples[bi]
        lab = ", ".join(f"{names[j]}={mp[j]:.3g}" for j in range(num_vars))
        reps = all_ts[bi]
        patched = Patch()
        p05 = p05_all[bi][::stride]; p95 = p95_all[bi][::stride]
        patched["data"][IDX_BAND95]["y"] = np.concatenate([p95, p05[::-1]]).tolist()
        p25 = p25_all[bi][::stride]; p75 = p75_all[bi][::stride]
        patched["data"][IDX_BAND75]["y"] = np.concatenate([p75, p25[::-1]]).tolist()
        patched["data"][IDX_MEDIAN]["y"] = med_all[bi][::stride].tolist()
        for ri in range(n_spaghetti):
            patched["data"][IDX_SPAG_START + ri]["y"] = reps[ri][::stride].tolist()
        patched["data"][IDX_MU]["y"] = mean_ts[bi][::stride].tolist()
        if thr_value is not None and thr_value != "":
            patched["data"][IDX_THRESHOLD]["y"] = [float(thr_value), float(thr_value)]
            patched["data"][IDX_THRESHOLD]["visible"] = True
            patched["data"][IDX_THRESHOLD]["showlegend"] = True
        else:
            patched["data"][IDX_THRESHOLD]["visible"] = False
            patched["data"][IDX_THRESHOLD]["showlegend"] = False
        patched["layout"]["title"]["text"] = f"Simulation \u2014 nearest: {lab}"
        return patched

    # --- Threshold panel: threshold + time window + conditioning sliders ---
    heatmap_outputs = [Output(f"heatmap-{pidx}","figure")
                       for pidx in range(len(pair_keys))]

    @app.callback(
        [Output("threshold-plot","figure"),
         Output("prim-text","children"),
         Output("cond-status","children")] + heatmap_outputs,
        [Input("threshold-input","value"),
         Input("time-window-slider","value")] +
        [Input(f"cond-slider-{n}","value") for n in names])
    def update_threshold_panel(thr_value, time_range, *cond_ranges):
        # Slider values are integer indices (0..n_ts-1).
        t_start = max(0, min(int(time_range[0]), n_ts - 1))
        t_end   = max(t_start, min(int(time_range[1]), n_ts - 1))

        # Parse conditioning ranges into dict: var_index -> (lo, hi)
        cond_bounds = {}
        for vi, rng in enumerate(cond_ranges):
            cond_bounds[vi] = (float(rng[0]), float(rng[1]))

        # --- No threshold: placeholders ---
        if thr_value is None or thr_value == "":
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No threshold set. Enter a value above.",
                xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
                font=dict(size=16,color="#999"))
            empty_fig.update_layout(height=300,template="plotly_white",
                xaxis=dict(visible=False),yaxis=dict(visible=False))
            no_msg = html.P("No threshold set.",
                style={"textAlign":"center","color":"#999","padding":"20px"})
            empty_hm = go.Figure()
            empty_hm.update_layout(height=380,template="plotly_white",
                xaxis=dict(visible=False),yaxis=dict(visible=False))
            return [empty_fig, no_msg, ""] + [empty_hm]*len(pair_keys)

        thr = float(thr_value)

        # --- Recompute exceedance + PAWN if threshold changed ---
        if _cache["threshold"] != thr:
            print(f"  Recomputing exceedance for c={thr} ...")
            _cache["exceed_ts"] = np.mean(all_ts > thr, axis=1)
            print("  Recomputing PAWN on exceedance ...")
            _cache["pawn_exceed"] = _pawn_at_each_timestep(
                prob, param_samples, _cache["exceed_ts"], n_ts)
            _cache["threshold"] = thr
            print("  Done.")

        et = _cache["exceed_ts"]
        pe = _cache["pawn_exceed"]

        # --- Build global conditioning mask (all variables) for PRIM ---
        global_mask = np.ones(len(param_samples), dtype=bool)
        cond_labels = []
        for vi in range(num_vars):
            lo, hi = cond_bounds[vi]
            is_full = (lo <= samp_mins[vi] + 1e-9) and (hi >= samp_maxs[vi] - 1e-9)
            if not is_full:
                global_mask &= (param_samples[:, vi] >= lo) & (param_samples[:, vi] <= hi)
                cond_labels.append(f"{names[vi]}\u2208[{lo:.2g},{hi:.2g}]")
        n_global = int(global_mask.sum())

        if cond_labels:
            status = f"Conditioning: {', '.join(cond_labels)} \u2014 {n_global} combinations"
        else:
            status = f"Full range (no conditioning) \u2014 {len(param_samples)} combinations"

        # --- Exceedance + PAWN figure (always uses full data) ---
        tfig = make_subplots(rows=2,cols=1,vertical_spacing=0.12,
            row_heights=[0.45,0.55],
            subplot_titles=(
                f"Exceedance Probability  P(Y > {thr})",
                f"PAWN (median) on P(Y > {thr})"))
        ci_idx_thr = []
        em=np.mean(et,axis=0); emd=np.median(et,axis=0)
        e10=np.percentile(et,10,axis=0); e90=np.percentile(et,90,axis=0)
        ci0 = len(tfig.data)
        tfig.add_trace(go.Scatter(
            x=np.concatenate([timesteps,timesteps[::-1]]),
            y=np.concatenate([e90,e10[::-1]]),fill="toself",
            fillcolor="rgba(220,60,60,0.15)",line=dict(width=0),
            name="10th\u201390th %ile"), row=1,col=1)
        ci_idx_thr.append(ci0)
        tfig.add_trace(go.Scatter(x=timesteps,y=em,
            line=dict(color="#DC3C3C",width=3),name="Mean P(exceed)"), row=1,col=1)
        tfig.add_trace(go.Scatter(x=timesteps,y=emd,
            line=dict(color="#DC3C3C",width=2,dash="dash"),
            name="Median P(exceed)"), row=1,col=1)
        pmed_e=pe["median"]; pmn_e=pe["minimum"]; pmx_e=pe["maximum"]
        for i,n in enumerate(names):
            c=colors[i%len(colors)]
            tfig.add_trace(go.Scatter(x=timesteps,y=pmed_e[n].values,
                mode="lines",name=f"PAWN({n})",line=dict(width=2.5,color=c),
                legendgroup=f"PE_{n}"), row=2,col=1)
            ci_k = len(tfig.data)
            tfig.add_trace(go.Scatter(
                x=np.concatenate([timesteps,timesteps[::-1]]),
                y=np.concatenate([pmx_e[n].values,pmn_e[n].values[::-1]]),
                fill="toself",fillcolor=_fill(c,0.15),line=dict(width=0),
                showlegend=False,legendgroup=f"PE_{n}",hoverinfo="skip"),
                row=2,col=1)
            ci_idx_thr.append(ci_k)
        ts_start_val = float(timesteps[t_start])
        ts_end_val   = float(timesteps[t_end])
        _ts_lbl = lambda v: str(int(v)) if v == int(v) else f"{v:.4g}"
        ts_start_lbl = _ts_lbl(ts_start_val)
        ts_end_lbl   = _ts_lbl(ts_end_val)
        for row in [1, 2]:
            tfig.add_vrect(x0=ts_start_val, x1=ts_end_val,
                fillcolor="rgba(178,34,34,0.08)",line_width=0,row=row,col=1)
            tfig.add_vline(x=ts_start_val, line_dash="dot", line_color="#B22222",
                           line_width=1,row=row,col=1)
            tfig.add_vline(x=ts_end_val, line_dash="dot", line_color="#B22222",
                           line_width=1,row=row,col=1)
        nt = len(tfig.data)
        va_t = [True]*nt; vn_t = [True]*nt
        for idx in ci_idx_thr: vn_t[idx] = False
        tfig.update_layout(height=700,template="plotly_white",
            legend=dict(orientation="v",y=0.5,yanchor="middle",
                        x=1.02,xanchor="left",font=dict(size=10)),
            margin=dict(t=40,b=40,l=60,r=120),hovermode="x unified",
            updatemenus=[dict(type="buttons",direction="left",
                x=0.0,xanchor="left",y=1.04,yanchor="bottom",
                buttons=[
                    dict(label="\u2611 CI",method="update",args=[{"visible":va_t}]),
                    dict(label="\u2610 CI",method="update",args=[{"visible":vn_t}]),
                ], font=dict(size=11),bgcolor="white",bordercolor="#ccc")])
        tfig.update_yaxes(title_text="P(Y>c)",row=1,col=1,range=[0,1.05])
        tfig.update_yaxes(title_text="PAWN median",row=2,col=1,range=[0,1.05])
        tfig.update_xaxes(title_text="Series",row=2,col=1)

        # --- Max-over-window exceedance (for heatmaps) ---
        window_exceed = et[:, t_start:t_end+1]
        max_exceed = window_exceed.max(axis=1)

        # --- PRIM on conditioned subset ---
        prim_mask = global_mask if cond_labels else None
        prim_res = _prim_for_window(param_samples, et, names, t_start, t_end,
                                    sample_mask=prim_mask)

        if prim_res is not None:
            box = prim_res["best_box"]
            prim_lines = [
                f"PRIM Danger Zone  [t={ts_start_lbl}\u2013{ts_end_lbl}]  c={thr}"]
            if cond_labels:
                prim_lines.append(f"Conditioned: {', '.join(cond_labels)}")
            prim_lines.append(
                f"density={box['density']:.1%}  coverage={box['coverage']:.1%}  "
                f"support={box['support']:.1%}  n={box['n_in_box']}\n")
            for k, (lo, hi) in enumerate(box["bounds"]):
                orig_lo, orig_hi = actual_bounds[k]
                tag = ""
                if lo > orig_lo + 1e-6 or hi < orig_hi - 1e-6:
                    tag = " \u25c0 restricted"
                prim_lines.append(f"  {names[k]:>8s}: [{lo:.3f}, {hi:.3f}]{tag}")
            prim_lines.append(f"\nPeeling ({len(prim_res['trajectory'])} steps):")
            prim_lines.append(
                f"  {'#':>3s}  {'Dens':>7s}  {'Cover':>7s}  {'Supp':>7s}  {'n':>6s}")
            for si, step in enumerate(prim_res["trajectory"]):
                prim_lines.append(
                    f"  {si:>3d}  {step['density']:>7.1%}  {step['coverage']:>7.1%}  "
                    f"{step['support']:>7.1%}  {step['n_in_box']:>6d}")
            prim_text = "\n".join(prim_lines)
        else:
            box = None
            n_cond = n_global if cond_labels else len(param_samples)
            prim_text = (f"PRIM  [t={ts_start_lbl}\u2013{ts_end_lbl}]  c={thr}: insufficient "
                         f"exceedance to fit a box (n={n_cond} conditioned samples).")

        prim_div = html.Details([
            html.Summary(f"PRIM Danger Zone  [t={ts_start_lbl}\u2013{ts_end_lbl}]",
                         style={"fontWeight":"bold","fontSize":"13px",
                                "cursor":"pointer","color":"#B22222"}),
            html.Pre(prim_text, style={
                "backgroundColor":"#fef8f8","padding":"12px",
                "borderRadius":"6px","border":"1px solid #dcc",
                "fontSize":"11px","lineHeight":"1.4",
                "maxHeight":"400px","overflowY":"auto","whiteSpace":"pre-wrap"}),
        ], style={"margin":"8px 5px"}, open=True)

        # --- Heatmaps: per-pair conditioning ---
        hfigs = []
        for pidx, (ni, nj) in enumerate(pair_keys):
            idx_i = names.index(ni); idx_j = names.index(nj)

            # Build mask for this pair: condition on all vars EXCEPT the pair axes
            pair_mask = np.ones(len(param_samples), dtype=bool)
            pair_cond_labels = []
            for vi in range(num_vars):
                if vi == idx_i or vi == idx_j:
                    continue  # don't condition on the pair's own axes
                lo, hi = cond_bounds[vi]
                is_full = (lo <= samp_mins[vi] + 1e-9) and (hi >= samp_maxs[vi] - 1e-9)
                if not is_full:
                    pair_mask &= (param_samples[:, vi] >= lo) & (param_samples[:, vi] <= hi)
                    pair_cond_labels.append(f"{names[vi]}\u2208[{lo:.2g},{hi:.2g}]")

            use_mask = pair_mask if pair_cond_labels else None
            hdata = _heatmap_single_pair(param_samples, max_exceed,
                                         idx_i, idx_j, sample_mask=use_mask)

            if hdata is None:
                hfig = go.Figure()
                n_avail = int(pair_mask.sum()) if pair_cond_labels else len(param_samples)
                hfig.add_annotation(
                    text=f"Too few samples (n={n_avail}).<br>Widen conditioning bounds.",
                    xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
                    font=dict(size=13,color="#B22222"))
                hfig.update_layout(
                    title=f"P(exceed) \u2014 {ni} vs {nj}  [INSUFFICIENT DATA]",
                    template="plotly_white", height=380,
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    margin=dict(t=40,b=40,l=60,r=20))
            else:
                subtitle = ""
                if pair_cond_labels:
                    subtitle = f"  |  {', '.join(pair_cond_labels)}"
                    subtitle += f"  (n={hdata['n_samples']})"
                hfig = go.Figure(data=go.Heatmap(
                    z=hdata["grid"],
                    x=np.round(hdata["x_mids"],3),
                    y=np.round(hdata["y_mids"],3),
                    colorscale="RdYlGn_r", zmin=0, zmax=1,
                    colorbar=dict(title="P(exc)", len=0.8),
                    hovertemplate=f"{ni}=%{{x}}<br>{nj}=%{{y}}<br>"
                                  f"P(exceed)=%{{z:.2f}}<extra></extra>"))
                if box is not None:
                    hfig.add_shape(type="rect",
                        x0=box["bounds"][idx_i][0], x1=box["bounds"][idx_i][1],
                        y0=box["bounds"][idx_j][0], y1=box["bounds"][idx_j][1],
                        line=dict(color="black",width=2,dash="dash"),
                        fillcolor="rgba(0,0,0,0)")
                hfig.update_layout(
                    title=f"P(exceed) \u2014 {ni} vs {nj}  [t={ts_start_lbl}\u2013{ts_end_lbl}]{subtitle}",
                    xaxis_title=ni, yaxis_title=nj,
                    template="plotly_white", height=380,
                    margin=dict(t=40,b=40,l=60,r=20))
            hfigs.append(hfig)

        return [tfig, prim_div, status] + hfigs

    return app


