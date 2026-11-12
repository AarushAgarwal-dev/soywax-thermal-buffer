#!/usr/bin/env python3
"""Soy-wax phase-change thermal buffer for a compact AI chip: simulation prototype.

Thermal network for one hardware build, stepped explicitly in time:

    P(t) -> [junction + package] --R_jc--> [heatsink] --R_ha--> air (fan)
                                               |
                                           R_contact
                                               |
                                 [wax 1]--[wax 2]-- ... --[wax N]   (far face insulated)

* The wax pack is a 1-D slab split into N layers. Each layer tracks its enthalpy, so
  melting and re-solidification over soy wax's broad melting range fall out of the
  energy balance. Layer temperature and melt fraction come from a lookup table.
* Natural convection inside molten wax is ignored. That is conservative for pure wax.
* The chip runs AI jobs that each need a fixed amount of work. Above the throttle limit
  an integral DVFS governor lowers the clock scale s. Power is P_static + P_dyn * s^3
  and work progresses at rate s, so throttling makes jobs take longer.

Run `python soywax_sim.py --help` for the knobs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

AL_RHO = 2700.0          # kg/m^3, aluminum
AL_CP = 900.0            # J/(kg K), aluminum
THROTTLED_BELOW = 0.995  # clock scale below this counts as throttled
T_END = 2160.0           # s, length of the default scenario


@dataclass(frozen=True)
class WaxProps:
    """Container-blend soy wax (hydrogenated soybean oil). Sources in README.md."""
    t_melt: float = 49.0       # degC, centre of the melting range
    melt_sigma: float = 3.0    # degC, spread: 95 % of the melting happens within +/- 2 sigma
    latent: float = 110e3      # J/kg
    cp_solid: float = 2000.0   # J/(kg K)
    cp_liquid: float = 2200.0  # J/(kg K)
    k_solid: float = 0.17      # W/(m K)
    k_liquid: float = 0.15     # W/(m K)
    rho: float = 900.0         # kg/m^3 (solid), used to size the pack


@dataclass(frozen=True)
class Device:
    """Compact edge-AI module: 30 W boost, 4 W idle, small fan heatsink."""
    t_amb: float = 25.0     # degC, air entering the heatsink
    p_idle: float = 4.0     # W between jobs
    p_static: float = 3.0   # W, leakage and uncore while a job runs
    p_dyn: float = 27.0     # W, dynamic power at full clock (full speed = 30 W)
    c_j: float = 4.0        # J/K, die + package + nearby board
    r_jc: float = 0.7       # K/W, junction -> heatsink base incl. thermal paste
    c_hs: float = 25.0      # J/K, heatsink (about 28 g of aluminum)
    r_ha: float = 2.4       # K/W, heatsink -> air with a small fan
    t_limit: float = 85.0   # degC, throttle setpoint
    s_min: float = 0.3      # lowest clock scale the governor may choose
    k_gov: float = 0.006    # 1/(K s), integral gain of the DVFS governor

    @property
    def p_full(self) -> float:
        return self.p_static + self.p_dyn


@dataclass(frozen=True)
class Build:
    key: str
    label: str
    wax_mass: float = 0.0        # kg of soy wax in the pack
    wax_fraction: float = 1.0    # wax share of the pack volume (1.0 = pure wax)
    k_pack: float | None = None  # W/(m K) effective pack conductivity; None = pure wax
    extra_al: float = 0.0        # kg of aluminum added to the heatsink (control case)
    area: float = 50e-3 * 50e-3  # m^2, pack footprint on the heatsink
    h_contact: float = 3000.0    # W/(m^2 K), heatsink <-> pack interface
    n_layers: int = 12


@dataclass(frozen=True)
class Job:
    name: str
    arrival: float       # s
    work: float          # s of work at full clock
    kind: str = "burst"  # "burst" or "sustained"


def default_jobs() -> list[Job]:
    """Three 90 s inference bursts, 7 min apart, then an 8 min sustained job."""
    return [
        Job("Job 1", 60.0, 90.0),
        Job("Job 2", 480.0, 90.0),
        Job("Job 3", 900.0, 90.0),
        Job("Sustained job", 1320.0, 480.0, kind="sustained"),
    ]


def default_builds(wax_g: float = 20.0, k_foam: float = 3.0, foam_wax_fraction: float = 0.92) -> list[Build]:
    m = wax_g / 1000.0
    return [
        Build("stock", "Stock heatsink"),
        Build("wax", f"+{wax_g:g} g soy wax", wax_mass=m),
        Build("foam", f"+{wax_g:g} g soy wax in aluminum foam", wax_mass=m,
              wax_fraction=foam_wax_fraction, k_pack=k_foam),
        Build("alu", f"+{wax_g:g} g aluminum (control)", extra_al=m),
    ]


_erf = np.vectorize(math.erf, otypes=[float])


class LayerTable:
    """Energy <-> temperature lookup for one pack layer (wax + optional metal matrix)."""

    def __init__(self, wax: WaxProps, m_wax: float, m_metal: float,
                 t_lo: float = -20.0, t_hi: float = 200.0, step: float = 0.005):
        T = np.arange(t_lo, t_hi + 0.5 * step, step)
        phi = 0.5 * (1.0 + _erf((T - wax.t_melt) / (wax.melt_sigma * math.sqrt(2.0))))
        cp = wax.cp_solid + (wax.cp_liquid - wax.cp_solid) * phi
        sensible = np.concatenate(([0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(T))))
        self.T = T
        self.phi = phi  # liquid (melted) fraction of the wax
        self.E = m_wax * (sensible + wax.latent * phi) + m_metal * AL_CP * (T - t_lo)


@dataclass
class Result:
    build: Build
    jobs: list[Job]
    t: np.ndarray            # s
    power: np.ndarray        # W
    clock: np.ndarray        # clock scale 0..1
    busy: np.ndarray         # a job is running
    t_j: np.ndarray          # degC junction
    t_hs: np.ndarray         # degC heatsink
    melt: np.ndarray         # melted share of the wax 0..1 (zeros without a pack)
    layer_melt: np.ndarray | None  # (samples, layers), layer 0 touches the heatsink
    job_start: list[float]
    job_end: list[float]
    dt: float
    pack: dict | None


def simulate(build: Build, device: Device = Device(), wax: WaxProps = WaxProps(),
             jobs: list[Job] | None = None, t_end: float = T_END, sample_dt: float = 0.5) -> Result:
    jobs = sorted(jobs if jobs is not None else default_jobs(), key=lambda j: j.arrival)
    c_hs = device.c_hs + build.extra_al * AL_CP
    g_jc = 1.0 / device.r_jc
    g_ha = 1.0 / device.r_ha
    # start from the idle steady state
    t_hs = device.t_amb + device.p_idle * device.r_ha
    t_j = t_hs + device.p_idle * device.r_jc

    pack = None
    if build.wax_mass > 0:
        n = build.n_layers
        v_pack = build.wax_mass / wax.rho / build.wax_fraction
        thickness = v_pack / build.area
        dx = thickness / n
        m_wax_i = build.wax_mass / n
        m_metal_i = AL_RHO * (1.0 - build.wax_fraction) * v_pack / n
        tab = LayerTable(wax, m_wax_i, m_metal_i)
        pure = build.k_pack is None
        r_contact = 1.0 / (build.h_contact * build.area)
        half = 0.5 * dx / build.area          # divide by k to get half-layer resistance
        e = np.full(n, float(np.interp(t_hs, tab.T, tab.E)))
        tw = np.full(n, t_hs)
        phi = np.interp(tw, tab.T, tab.phi)
        k_const = None if pure else np.full(n, float(build.k_pack))
        k_max = max(wax.k_solid, wax.k_liquid) if pure else float(build.k_pack)
        g_layer = k_max * build.area / dx
        g0 = 1.0 / (r_contact + half / k_max)
        c_layer = m_wax_i * wax.cp_solid + m_metal_i * AL_CP
        dt_lim = min(c_layer / (2 * g_layer), c_layer / (g0 + g_layer),
                     c_hs / (g_jc + g_ha + g0), device.c_j / g_jc)
        pack = dict(thickness_mm=thickness * 1e3, layer_mm=dx * 1e3,
                    metal_g=m_metal_i * n * 1e3, latent_J=build.wax_mass * wax.latent)
    else:
        n = 0
        dt_lim = min(c_hs / (g_jc + g_ha), device.c_j / g_jc)

    dt = min(0.05, 0.5 * dt_lim)
    every = max(1, math.ceil(sample_dt / dt))
    dt = sample_dt / every
    n_steps = int(round(t_end / dt))
    n_samp = n_steps // every + 1

    rec_t = np.empty(n_samp)
    rec_p = np.empty(n_samp)
    rec_s = np.empty(n_samp)
    rec_busy = np.zeros(n_samp, dtype=bool)
    rec_tj = np.empty(n_samp)
    rec_ths = np.empty(n_samp)
    rec_melt = np.zeros(n_samp)
    rec_layers = np.zeros((n_samp, n)) if pack else None

    start = [math.nan] * len(jobs)
    end = [math.nan] * len(jobs)
    queue: list[int] = []
    nxt, active, remaining = 0, -1, 0.0
    s = 1.0
    i_rec = 0
    for step in range(n_steps + 1):
        t = step * dt
        while nxt < len(jobs) and jobs[nxt].arrival <= t + 1e-9:
            queue.append(nxt)
            nxt += 1
        if active < 0 and queue:
            active = queue.pop(0)
            remaining = jobs[active].work
            start[active] = t
        # DVFS governor: integral action on the junction temperature error
        s -= device.k_gov * (t_j - device.t_limit) * dt
        s = 1.0 if s > 1.0 else (device.s_min if s < device.s_min else s)
        busy = active >= 0
        p = device.p_static + device.p_dyn * s ** 3 if busy else device.p_idle

        if step % every == 0:
            rec_t[i_rec], rec_p[i_rec], rec_s[i_rec] = t, p, s
            rec_busy[i_rec], rec_tj[i_rec], rec_ths[i_rec] = busy, t_j, t_hs
            if pack:
                rec_layers[i_rec] = phi
                rec_melt[i_rec] = phi.mean()
            i_rec += 1
        if step == n_steps:
            break

        if busy:
            remaining -= s * dt
            if remaining <= 0.0:
                end[active] = t + dt
                active = -1

        q_jc = (t_j - t_hs) * g_jc
        q_ha = (t_hs - device.t_amb) * g_ha
        if pack:
            k = wax.k_solid + (wax.k_liquid - wax.k_solid) * phi if pure else k_const
            r_half = half / k
            q0 = (t_hs - tw[0]) / (r_contact + r_half[0])
            q_int = (tw[:-1] - tw[1:]) / (r_half[:-1] + r_half[1:])
            de = np.zeros(n)
            de[0] = q0
            de[:-1] -= q_int
            de[1:] += q_int
            e += de * dt
            tw = np.interp(e, tab.E, tab.T)
            phi = np.interp(tw, tab.T, tab.phi)
        else:
            q0 = 0.0
        t_j += (p - q_jc) * dt / device.c_j
        t_hs += (q_jc - q_ha - q0) * dt / c_hs

    return Result(build, jobs, rec_t, rec_p, rec_s, rec_busy, rec_tj, rec_ths, rec_melt,
                  rec_layers, start, end, dt, pack)


def run_all(device: Device, wax: WaxProps, builds: list[Build], jobs: list[Job],
            t_end: float = T_END) -> list[Result]:
    return [simulate(b, device, wax, jobs, t_end) for b in builds]


def throttled_mask(res: Result) -> np.ndarray:
    return res.busy & (res.clock < THROTTLED_BELOW)


def summarize(res: Result) -> dict:
    t = res.t
    step = t[1] - t[0]
    jobs = res.jobs
    thr = throttled_mask(res)
    bursts = [i for i, j in enumerate(jobs) if j.kind == "burst"]
    sustained = [i for i, j in enumerate(jobs) if j.kind == "sustained"]
    out: dict = {"key": res.build.key, "build": res.build.label}
    if bursts:
        w_end = jobs[sustained[0]].arrival if sustained else t[-1] + step
        win = t < w_end
        out["peak_tj_bursts_C"] = float(res.t_j[win].max())
        rises = []
        for i in bursts:
            in_job = (t >= jobs[i].arrival) & (t <= res.job_end[i])
            rises.append(res.t_j[in_job].max() - np.interp(jobs[i].arrival, t, res.t_j))
        out["rise_per_burst_K"] = float(np.mean(rises))
        out["throttled_bursts_s"] = float(thr[win].sum() * step)
        times = [res.job_end[i] - jobs[i].arrival for i in bursts]
        out["burst_job_times_s"] = times
        out["burst_job_time_s"] = float(np.mean(times))
        out["burst_job_work_s"] = float(np.mean([jobs[i].work for i in bursts]))
    if sustained:
        i = sustained[0]
        st, en = res.job_start[i], res.job_end[i]
        stop = en if math.isfinite(en) else t[-1]
        win = (t >= st) & (t <= stop)
        idx = np.flatnonzero(thr & win)
        out["sustained_full_speed_s"] = float(t[idx[0]] - st) if idx.size else float(stop - st)
        out["sustained_job_time_s"] = float(en - jobs[i].arrival) if math.isfinite(en) else math.nan
        out["sustained_throttled_s"] = float(thr[win].sum() * step)
        out["peak_tj_sustained_C"] = float(res.t_j[win].max())
    if res.layer_melt is not None:
        out["peak_melt_pct"] = float(100 * res.melt.max())
        out["melt_left_at_next_job_pct"] = [float(100 * np.interp(j.arrival, t, res.melt)) for j in jobs[1:]]
        out["pack"] = res.pack
    return out


def melt_reach_mm(wax: WaxProps, seconds: float, superheat: float = 10.0) -> float:
    """Depth of pure wax melted in `seconds` when the hot face sits `superheat` K above the melt point
    (quasi-steady Stefan estimate, conduction only)."""
    return 1e3 * math.sqrt(2 * wax.k_liquid * superheat * seconds / (wax.rho * wax.latent))


def design_checks(device: Device, wax: WaxProps, wax_g: float, burst_s: float) -> list[str]:
    """Back-of-envelope rules that decide whether a wax buffer can help at all."""
    lo, hi = wax.t_melt - 2 * wax.melt_sigma, wax.t_melt + 2 * wax.melt_sigma
    hs_idle = device.t_amb + device.p_idle * device.r_ha
    hs_throttle = device.t_limit - device.p_full * device.r_jc
    p_reject = (wax.t_melt - device.t_amb) / device.r_ha
    latent = wax_g / 1000.0 * wax.latent
    burst_j = max(0.0, device.p_full - p_reject) * burst_s
    refreeze_w = p_reject - device.p_idle
    idle_needed = burst_j / refreeze_w if refreeze_w > 0 else math.inf
    depth_mm = melt_reach_mm(wax, burst_s)
    ok = "OK" if hs_idle < lo and hi < hs_throttle else "MISMATCH"
    return [
        f"Melting window : wax melts at {lo:.0f}-{hi:.0f} C; heatsink idles at {hs_idle:.1f} C and the "
        f"chip throttles once the heatsink passes {hs_throttle:.1f} C at full power -> {ok}",
        f"Latent capacity: {latent:.0f} J stored vs ~{burst_j:.0f} J of heat the fan cannot remove "
        f"during a {burst_s:.0f} s burst -> " + ("OK" if latent > 1.2 * burst_j else "pack too small"),
        f"Refreeze       : at the melt point the fan removes {p_reject:.1f} W, {refreeze_w:.1f} W above idle "
        f"power -> needs ~{idle_needed / 60:.1f} min idle after each burst to re-solidify",
        f"Pure-wax reach : {burst_s:.0f} s at 10 K above the melt point melts only ~{depth_mm:.1f} mm "
        f"of pure wax -> keep every bit of wax within ~{max(1, round(depth_mm))} mm of metal",
    ]


def mmss(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def print_table(summaries: list[dict]) -> None:
    head = (f"{'Build':36s}{'Peak Tj':>9s}{'Rise per':>10s}{'Throttled':>11s}{'Burst job':>11s}"
            f"{'Full speed':>12s}{'Sust. job':>11s}")
    sub = (f"{'':36s}{'(bursts)':>9s}{'burst':>10s}{'(bursts)':>11s}{'avg time':>11s}"
           f"{'(sustained)':>12s}{'time':>11s}")
    print(head)
    print(sub)
    for s in summaries:
        print(f"{s['build']:36s}{s['peak_tj_bursts_C']:>7.1f} C{s['rise_per_burst_K']:>8.1f} K"
              f"{s['throttled_bursts_s']:>9.0f} s{mmss(s['burst_job_time_s']):>11s}"
              f"{mmss(s['sustained_full_speed_s']):>12s}{mmss(s['sustained_job_time_s']):>11s}")


COLORS = {"stock": "#8b949e", "wax": "#e8a33d", "foam": "#22a99e", "alu": "#8957e5"}


def plot_summary(results: list[Result], device: Device, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.7, 1]))
    for r in results:
        m = r.t / 60
        c = COLORS.get(r.build.key, "k")
        ls = ":" if r.build.key == "alu" else "-"
        axs[0].plot(m, r.power, color=c, lw=1.3, ls=ls, label=r.build.label)
        axs[1].plot(m, r.t_j, color=c, lw=1.6, ls=ls, label=r.build.label)
        if r.layer_melt is not None:
            axs[2].plot(m, 100 * r.melt, color=c, lw=1.6, label=r.build.label)
    for j in results[0].jobs:
        for ax in axs:
            ax.axvspan(j.arrival / 60, (j.arrival + j.work) / 60, color="#d0d7de", alpha=0.35, lw=0)
        axs[0].text((j.arrival + j.work / 2) / 60, device.p_full * 1.08, j.name, ha="center", fontsize=8)
    axs[1].axhline(device.t_limit, color="#cf222e", ls="--", lw=1, label="throttle limit")
    axs[0].set_ylabel("Chip power (W)")
    axs[0].set_ylim(0, device.p_full * 1.2)
    axs[1].set_ylabel("Junction temperature (°C)")
    axs[2].set_ylabel("Wax melted (%)")
    axs[2].set_ylim(-3, 103)
    axs[2].set_xlabel("Time (min)")
    axs[1].legend(loc="upper left", fontsize=9)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("Soy-wax thermal buffer on a compact AI chip (simulation)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_advantages(results: list[Result], device: Device, path: Path) -> None:
    """Three panels: the first AI job, time throttled in the short jobs, time at full speed in the long job."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, panel, grid, text, muted, red = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e", "#ff6b6b"
    colors = {"stock": "#c9d1d9", "wax": "#f0a63a", "foam": "#2dd4bf"}
    R = {r.build.key: r for r in results}
    S = {k: summarize(R[k]) for k in colors}
    job = R["stock"].jobs[0]

    fig = plt.figure(figsize=(15, 5.4), dpi=120, facecolor=bg)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.22, left=0.055, right=0.985, top=0.80, bottom=0.15)
    ax1, ax2, ax3 = (fig.add_subplot(g) for g in gs)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=11)

    for k, c in colors.items():
        r = R[k]
        m = (r.t >= job.arrival - 10) & (r.t <= job.arrival + job.work + 25)
        ax1.plot(r.t[m] - job.arrival, r.t_j[m], color=c, lw=2.8, label=r.build.label)
    ax1.axhline(device.t_limit, color=red, ls="--", lw=1.3)
    ax1.text(job.work + 24, device.t_limit + 1.3, f"throttle limit {device.t_limit:.0f} °C", color=red, ha="right", fontsize=10)
    ax1.axvspan(0, job.work, color="white", alpha=0.05, lw=0)
    ax1.text(job.work / 2, 90.6, "AI job running", color=muted, ha="center", va="center", fontsize=10)
    ax1.set_ylim(34, 93)
    ax1.set_xlim(-10, job.work + 25)
    ax1.set_xlabel("Seconds since the job started", color=text, fontsize=12)
    ax1.set_ylabel("Chip temperature (°C)", color=text, fontsize=12)
    ax1.grid(color=grid, alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10.5, frameon=False, labelcolor=text)
    ax1.set_title(f"Chip temperature during the first {job.work:.0f}-second AI job", color=text, fontsize=13, loc="left")

    keys = list(colors)
    names = ["Stock\nheatsink", "+ soy wax", "+ soy wax in\naluminum foam"]
    nb = sum(1 for j in R["stock"].jobs if j.kind == "burst")
    long_min = next((j.work / 60 for j in R["stock"].jobs if j.kind == "sustained"), 0.0)
    art = "an" if str(int(round(long_min))).startswith("8") else "a"
    panels = ((ax2, [S[k]["throttled_bursts_s"] for k in keys], lambda v: f"{v:.0f} s",
               f"Time throttled in\n{nb} short jobs (lower is better)"),
              (ax3, [S[k]["sustained_full_speed_s"] for k in keys], mmss,
               f"Time at full speed in\n{art} {long_min:.0f}-minute job (higher is better)"))
    for ax, vals, fmt, title in panels:
        bars = ax.bar(range(3), vals, color=[colors[k] for k in keys], width=0.62)
        top = max(vals) or 1.0
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt(v), ha="center", va="bottom",
                    color=text, fontsize=14, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, color=text, fontsize=10)
        ax.set_yticks([])
        ax.set_ylim(0, top * 1.2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, color=text, fontsize=12, loc="left")
    fig.suptitle("Same chip, same fan, same workload: only the buffer on the heatsink changes (simulation)",
                 color=muted, fontsize=12, x=0.055, ha="left", y=0.97)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def build_scenario(args: argparse.Namespace) -> tuple[Device, WaxProps, list[Build], list[Job]]:
    device = replace(Device(), t_amb=args.t_amb, p_idle=args.p_idle,
                     p_dyn=args.p_full - Device().p_static, r_ha=args.r_ha, c_hs=args.c_hs)
    wax = replace(WaxProps(), t_melt=args.t_melt, latent=args.latent * 1e3)
    return device, wax, default_builds(args.wax_g, args.k_foam), default_jobs()


def add_scenario_args(ap: argparse.ArgumentParser) -> None:
    d, w = Device(), WaxProps()
    ap.add_argument("--wax-g", type=float, default=20.0, help="grams of soy wax in the pack (default 20)")
    ap.add_argument("--latent", type=float, default=w.latent / 1e3, help="latent heat, J/g (default 110)")
    ap.add_argument("--t-melt", type=float, default=w.t_melt, help="melting-range centre, C (default 49)")
    ap.add_argument("--k-foam", type=float, default=3.0, help="wax + aluminum foam conductivity, W/mK (default 3)")
    ap.add_argument("--t-amb", type=float, default=d.t_amb, help="air temperature, C (default 25)")
    ap.add_argument("--p-full", type=float, default=d.p_full, help="full-speed chip power, W (default 30)")
    ap.add_argument("--p-idle", type=float, default=d.p_idle, help="idle chip power, W (default 4)")
    ap.add_argument("--r-ha", type=float, default=d.r_ha, help="heatsink-to-air resistance, K/W (default 2.4)")
    ap.add_argument("--c-hs", type=float, default=d.c_hs, help="heatsink heat capacity, J/K (default 25)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(ap)
    ap.add_argument("--out", default="out", help="output folder (default ./out)")
    args = ap.parse_args(argv)

    device, wax, builds, jobs = build_scenario(args)
    results = run_all(device, wax, builds, jobs)
    summaries = [summarize(r) for r in results]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    burst_s = float(np.mean([j.work for j in jobs if j.kind == "burst"]))
    checks = design_checks(device, wax, args.wax_g, burst_s)

    print("Design checks")
    for line in checks:
        print("  " + line)
    print()
    print_table(summaries)
    for s in summaries:
        if "melt_left_at_next_job_pct" in s:
            left = ", ".join(f"{v:.0f}%" for v in s["melt_left_at_next_job_pct"])
            print(f"  {s['build']}: peak melt {s['peak_melt_pct']:.0f}%, still molten when next job arrives: {left}")

    payload = dict(device=asdict(device), wax=asdict(wax), jobs=[asdict(j) for j in jobs],
                   design_checks=checks, results=summaries)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    plot_summary(results, device, out / "summary.png")
    plot_advantages(results, device, out / "advantages.png")
    print(f"\nWrote {out / 'results.json'}, {out / 'summary.png'} and {out / 'advantages.png'}")


if __name__ == "__main__":
    main()
