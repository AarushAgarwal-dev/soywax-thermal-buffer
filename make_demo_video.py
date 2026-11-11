#!/usr/bin/env python3
"""Render the narrated soy-wax demo video (1920x1080, H.264 + AAC) from soywax_sim.py.

    python make_demo_video.py              # out/soywax_demo.mp4 with voiceover
    python make_demo_video.py --no-voice   # silent version with captions only (works offline)
    python make_demo_video.py --stills     # a few PNG frames for a quick layout check

The voiceover uses Microsoft Edge's online neural voices through the edge-tts package. It needs
internet the first time; clips are cached in out/voice/. The timeline is built around the
narration: playback slows wherever a line needs more time, so each line is heard while the event
it describes is on screen. Frames are rendered in parallel worker processes.

Accepts the same scenario options as soywax_sim.py (--wax-g, --latent, --t-amb, ...).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import soywax_sim as sim  # noqa: E402

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
RED = "#ff6b6b"
GREEN = "#3fb950"
COLORS = {"stock": "#c9d1d9", "wax": "#f0a63a", "foam": "#2dd4bf", "alu": "#a371f7"}
LETTERS = {"stock": "A", "wax": "B", "foam": "C", "alu": "D"}
WAX_SOLID = np.array(to_rgb("#efe6cf"))
WAX_LIQUID = np.array(to_rgb("#d4881c"))
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "temp", ["#2f6db5", "#8fb3d9", "#e8e2d0", "#f0a04b", "#e0452f", "#9e1b1b"])
TEMP_NORM = Normalize(30, 95)
SHOWN = ("stock", "wax", "foam")
OUTLINE = [pe.withStroke(linewidth=2.5, foreground=BG)]
HATCH_ALPHA = 0.45
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

VOICE = "en-US-AndrewNeural"
RATE = "+4%"
SR = 48000          # audio sample rate
GAP = 0.45          # s of silence kept between narration lines
INTRO_LEAD = 0.7    # s of title card before the voice starts
REVEAL_S = 0.4      # fade-in time of intro-card elements
WORDS_PER_S = 2.5   # pace estimate used for --no-voice timing
TITLE = "Soy-wax thermal buffer for a compact AI chip (simulation)"


class Frame(NamedTuple):
    t: float                   # simulated device time shown (s)
    speed: float               # playback speed (simulated s per video s)
    overlay: str | None = None  # "intro", "outro" or None
    alpha: float = 0.0         # overlay opacity
    cue: int = -1              # caption index, -1 = none
    cue_age: int = 0           # frames since the caption appeared
    intro_t: float = 0.0       # s since the intro card appeared


@dataclass(frozen=True)
class Cue:
    t: float    # simulated time at which the line starts
    text: str   # on-screen caption
    say: str    # spoken version


@dataclass(frozen=True)
class Clip:
    key: str
    text: str
    duration: float                                  # s of speech after trimming silence
    audio: np.ndarray | None = None                  # mono float at SR
    words: tuple[tuple[float, str], ...] = ()        # (start s, word) after trimming


# ---------------------------------------------------------------------------- small helpers
def setup_fonts() -> None:
    families = []
    for name in ("Segoe UI", "Segoe UI Symbol", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            families.append(name)
        except ValueError:
            pass
    plt.rcParams["font.family"] = families or ["DejaVu Sans"]
    plt.rcParams["hatch.linewidth"] = 0.7
    plt.rcParams["savefig.facecolor"] = BG


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def article(n: int) -> str:
    return "an" if str(n).startswith("8") or n in (11, 18) else "a"


def say_seconds(s: float) -> str:
    s = int(round(s))
    if s < 120:
        return f"{s} seconds"
    m, r = divmod(s, 60)
    return f"{m} minutes" if r == 0 else f"{m} minutes and {r} seconds"


def wrap2(text: str, width: int = 96) -> str:
    """Split a long caption into two balanced lines, preferring a break after punctuation."""
    if "\n" in text or len(text) <= width:
        return text
    mid = len(text) / 2
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - mid) - (14 if text[i - 1] in ".,:;" else 0))
    return text[:cut] + "\n" + text[cut + 1:]


def first_time(mask: np.ndarray, t: np.ndarray, t0: float, t1: float) -> float | None:
    idx = np.flatnonzero(mask & (t >= t0) & (t < t1))
    return float(t[idx[0]]) if idx.size else None


def window_max(arr: np.ndarray, t: np.ndarray, t0: float, t1: float) -> float:
    return float(arr[(t >= t0) & (t <= t1)].max())


def wax_color(melted: float) -> np.ndarray:
    return WAX_SOLID * (1 - melted) + WAX_LIQUID * melted


def draw_stack(ax, key: str, n_layers: int, size: float = 10) -> dict:
    """Side view: fan heatsink on the chip, wax pack under the heatsink extension."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    arts = []

    def add(a):
        arts.append(a)
        return a

    hs = [add(ax.add_patch(Rectangle((0.5, 4.85), 9.0, 0.7, ec="none")))]
    hs += [add(ax.add_patch(Rectangle((0.7 + i * 1.15, 5.55), 0.45, 3.0, ec="none"))) for i in range(8)]
    add(ax.text(5.0, 9.35, "↑   ↑   ↑    fan airflow    ↑   ↑   ↑", ha="center", va="center",
                fontsize=size, color=MUTED))
    if size >= 12:  # too cramped to label the heatsink base at card size
        add(ax.text(5.0, 5.2, "heatsink", ha="center", va="center", fontsize=size - 1, color="white",
                    path_effects=OUTLINE))
    add(ax.add_patch(Rectangle((0.5, 3.4), 5.0, 0.45, fc="#1f6f43", ec="none")))
    chip = add(ax.add_patch(Rectangle((1.3, 3.85), 3.0, 1.0, ec=BG, lw=0.8)))
    add(ax.text(2.8, 4.35, "AI chip", ha="center", va="center", fontsize=size, color="white",
                fontweight="bold", path_effects=OUTLINE))
    add(ax.text(3.0, 2.95, "board", ha="center", va="center", fontsize=size - 1, color=MUTED))
    x0, x1, y_top, y_bot = 5.9, 9.3, 4.85, 0.8
    layers, hatch = [], None
    if key == "stock":
        add(ax.add_patch(Rectangle((x0, y_bot), x1 - x0, y_top - y_bot, fill=False, ec=MUTED, lw=1, ls="--")))
        add(ax.text((x0 + x1) / 2, (y_top + y_bot) / 2, "no\nbuffer", ha="center", va="center",
                    fontsize=size + 1, color=MUTED))
    else:
        h = (y_top - y_bot) / n_layers
        layers = [add(ax.add_patch(Rectangle((x0, y_top - (i + 1) * h), x1 - x0, h, ec="none")))
                  for i in range(n_layers)]
        if key == "foam":
            hatch = add(ax.add_patch(Rectangle((x0, y_bot), x1 - x0, y_top - y_bot, fill=False,
                                               ec=(0.1, 0.1, 0.1, HATCH_ALPHA), lw=0, hatch="xx")))
        add(ax.add_patch(Rectangle((x0, y_bot), x1 - x0, y_top - y_bot, fill=False, ec="#c9d1d9", lw=1.2)))
        add(ax.text((x0 + x1) / 2, 0.3, "soy wax" if key == "wax" else "wax + aluminum foam", ha="center",
                    va="center", fontsize=size, color=MUTED))
    return dict(chip=chip, hs=hs, layers=layers, hatch=hatch, artists=arts)


# ---------------------------------------------------------------------------- the dashboard
class Demo:
    def __init__(self, results, summaries, device, wax, jobs):
        self.R = {r.build.key: r for r in results}
        self.S = {s["key"]: s for s in summaries}
        self.device, self.wax, self.jobs = device, wax, jobs
        self.shown = [self.R[k] for k in SHOWN]
        self.t_end = float(self.shown[0].t[-1])
        self.dt_rec = float(self.shown[0].t[1] - self.shown[0].t[0])
        self.thr_cum = {r.build.key: np.cumsum(sim.throttled_mask(r)) * self.dt_rec for r in results}
        self.bursts = [j for j in jobs if j.kind == "burst"]
        self.sustained = next((j for j in jobs if j.kind == "sustained"), None)
        self.cues = self._cues()
        self.cue_text = [wrap2(c.text) for c in self.cues]
        self.intro_say, self.intro_keys = self._intro_script()
        self.outro_say = self._outro_script()
        self.intro_reveal = [0.0] * (len(self.intro_keys) + 1)

        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        self._header()
        self.cards = [self._card(r, k) for k, r in enumerate(self.shown)]
        self._plots()
        self._caption_bar()
        self.intro = self._intro()
        self.outro = self._outro()

    # ------------------------------------------------------------------ script
    def _cues(self) -> list[Cue]:
        R, S, d, w = self.R, self.S, self.device, self.wax
        stock, pure, foam = R["stock"], R["wax"], R["foam"]
        t = stock.t
        t_idle = stock.t_j[0]
        cues = [Cue(0.0,
                    f"Each card on the left is one build. At idle ({d.p_idle:.0f} W) all three sit near "
                    f"{t_idle:.0f} °C and the wax is solid.",
                    f"Each card on the left is one build. At idle, all three sit near {t_idle:.0f} degrees "
                    "Celsius, and the wax is solid.")]
        if self.bursts:
            j1 = self.bursts[0]
            i1 = self.jobs.index(j1)
            cues.append(Cue(j1.arrival - 4,
                            f"{j1.name} arrives: the AI chip jumps from {d.p_idle:.0f} W to {d.p_full:.0f} W.",
                            f"Job one arrives, and the chip jumps from {d.p_idle:.0f} to {d.p_full:.0f} watts."))
            t_thr = first_time(sim.throttled_mask(stock), t, j1.arrival, j1.arrival + 3 * j1.work)
            if t_thr is not None:
                cues.append(Cue(t_thr,
                                f"The stock heatsink hits the {d.t_limit:.0f} °C limit, so the chip throttles: "
                                "it cuts its clock speed.",
                                f"The stock heatsink hits the {d.t_limit:.0f} degree limit, so the chip "
                                "throttles, cutting its clock speed."))
            pk = math.ceil(window_max(foam.t_j, t, j1.arrival, foam.job_end[i1]))
            cues.append(Cue((t_thr if t_thr is not None else j1.arrival + 30) + 26,
                            f"Both wax packs melt near {w.t_melt:.0f} °C and soak up heat.\n"
                            f"In aluminum foam, the wax holds the chip at or below {pk} °C.",
                            f"Both wax packs are now melting near {w.t_melt:.0f} degrees, soaking up heat. "
                            f"In aluminum foam, the wax holds the chip at or below {pk} degrees."))
            later = [j.arrival for j in self.jobs[i1 + 1:]]
            if later:
                base = np.interp(later, t, stock.t_j)
                diffs = np.concatenate([np.interp(later, t, R[k].t_j) - base for k in ("wax", "foam")])
                lo, hi = max(0, int(round(diffs.min()))), int(round(diffs.max()))
                cues.append(Cue(max(r.job_end[i1] for r in self.shown) + 35,
                                "Between jobs the wax re-solidifies and hands its heat to the fan,\n"
                                f"so the wax builds start each new job {lo}–{hi} °C warmer.",
                                "Between jobs, the wax re-solidifies and hands its heat to the fan, so the "
                                f"wax builds start each new job {lo} to {hi} degrees warmer."))
        if len(self.bursts) > 1:
            j2 = self.bursts[1]
            frac = window_max(pure.melt, t, j2.arrival, j2.arrival + j2.work + 60)
            thick = pure.pack["thickness_mm"]
            depth = frac * thick
            cues.append(Cue(j2.arrival - 4,
                            f"{j2.name}: the plain wax pack helps much less. Wax conducts heat poorly "
                            f"(~{w.k_solid:.2f} W/m·K),\nso only the first {depth:.1f} mm of its {thick:.0f} mm "
                            "slab melts.",
                            "Job two. The plain wax pack helps much less. Wax conducts heat poorly, so only the "
                            f"first {depth:.1f} millimeters of its {thick:.0f} millimeter slab melts."))
        if len(self.bursts) > 2:
            j3 = self.bursts[2]
            nb = len(self.bursts)
            per_job = S["stock"]["throttled_bursts_s"] / nb
            if S["foam"]["throttled_bursts_s"] == 0:
                text = (f"{j3.name}: across all {nb} short jobs the foam build never throttles.\n"
                        f"The stock heatsink throttles for about {per_job:.0f} s of every {j3.work:.0f} s job.")
                say = (f"Job three. Across all {NUM_WORDS.get(nb, nb)} short jobs, the foam build never "
                       f"throttles, while the stock heatsink throttles for about {per_job:.0f} seconds of "
                       f"every {j3.work:.0f}.")
            else:
                text = (f"{j3.name}: the foam build throttled {S['foam']['throttled_bursts_s']:.0f} s in total, "
                        f"the stock heatsink {S['stock']['throttled_bursts_s']:.0f} s.")
                say = (f"Job three. In total, the foam build throttled for "
                       f"{S['foam']['throttled_bursts_s']:.0f} seconds, the stock heatsink for "
                       f"{S['stock']['throttled_bursts_s']:.0f}.")
            cues.append(Cue(j3.arrival - 4, text, say))
        js = self.sustained
        if js is not None:
            i = self.jobs.index(js)
            m = round(js.work / 60)
            cues.append(Cue(js.arrival - 4, f"Now {article(m)} {m}-minute job at full load.",
                            f"Now, {article(m)} {m} minute job at full load."))
            ts = first_time(sim.throttled_mask(stock), t, js.arrival, self.t_end)
            if ts is not None:
                ss, sb = S["stock"]["sustained_full_speed_s"], S["wax"]["sustained_full_speed_s"]
                cues.append(Cue(ts,
                                f"The stock heatsink throttles after {sim.mmss(ss)} at full speed, "
                                f"the plain wax build after {sim.mmss(sb)}.",
                                f"The stock heatsink throttles after {say_seconds(ss)}, and the plain wax "
                                f"build after {say_seconds(sb)}."))
            tf = first_time(sim.throttled_mask(foam), t, js.arrival, self.t_end)
            if tf is not None:
                sf = S["foam"]["sustained_full_speed_s"]
                mf = float(np.interp(tf, t, foam.melt))
                if mf >= 0.97:
                    state = state_say = "fully melted"
                elif mf >= 0.85:
                    state = state_say = "almost fully melted"
                else:
                    state, state_say = f"{100 * mf:.0f} % melted", f"{100 * mf:.0f} percent melted"
                cues.append(Cue(tf,
                                f"After {sim.mmss(sf)} the foam pack is {state}. The buffer is spent, "
                                "so this build throttles too.",
                                f"After {say_seconds(sf)}, the foam pack is {state_say}. The buffer is spent, "
                                "so this build throttles too."))
                late = (t > (ts if ts is not None else js.arrival) + 120) & (t < stock.job_end[i] - 5)
                p_sus = float(np.median(stock.power[late])) if late.any() else d.p_full
                cues.append(Cue(tf + 90,
                                f"It settles to the same ~{p_sus:.0f} W as the stock heatsink:\n"
                                "wax buys time but adds no sustained cooling capacity.",
                                f"From here, it settles to the same {p_sus:.0f} watts as the stock heatsink. "
                                "Wax buys time, but adds no sustained cooling."))
            end = max(r.job_end[i] for r in self.shown if math.isfinite(r.job_end[i]))
            mins = (self.t_end - foam.job_end[i]) / 60
            if math.isfinite(mins) and mins > 1:
                pct = 100 * foam.melt[-1]
                cues.append(Cue(end + 20,
                                f"Re-solidifying after a long job is slow: {mins:.0f} minutes after it ends, "
                                f"the foam pack is still {pct:.0f} % molten.",
                                f"Re-solidifying after a long job is slow. {mins:.0f} minutes later, the foam "
                                f"pack is still {pct:.0f} percent molten."))
        cues = sorted((c for c in cues if c.t < self.t_end - 5), key=lambda c: c.t)
        fixed = [cues[0]]
        for c in cues[1:]:  # keep cue times strictly increasing
            fixed.append(replace(c, t=max(c.t, fixed[-1].t + 2.0)))
        return fixed

    def _intro_script(self) -> tuple[str, list[str]]:
        wax_g = self.R["wax"].build.wax_mass * 1e3
        nb = len(self.bursts)
        say = ("Can soy wax smooth out heat spikes in compact AI hardware? "
               "This simulation compares three builds of a small AI chip with a fan heatsink: "
               f"the stock heatsink, the same heatsink with {wax_g:.0f} grams of soy wax, "
               "and that wax cast into aluminum foam. "
               f"Each build runs {NUM_WORDS.get(nb, nb)} short AI jobs"
               + (", then a long one." if self.sustained else "."))
        # intro-card groups after the title: subtitle, rows A/B/C, workload line
        return say, ["simulation", "stock", "same", "that", "each"]

    def _outro_script(self) -> str:
        S = self.S
        st, wx, fm, al = (S[k] for k in ("stock", "wax", "foam", "alu"))
        nb = len(self.bursts)
        s = (f"The scorecard. Over the {NUM_WORDS.get(nb, nb)} short jobs, the stock heatsink throttled for "
             f"{st['throttled_bursts_s']:.0f} seconds. ")
        if wx["throttled_bursts_s"] >= al["throttled_bursts_s"]:
            cmp = "no better than"
        elif wx["throttled_bursts_s"] >= 0.75 * al["throttled_bursts_s"]:
            cmp = "about the same as"
        else:
            cmp = "better than"
        s += f"Plain wax cut that to {wx['throttled_bursts_s']:.0f}, {cmp} the same weight of plain aluminum. "
        if fm["throttled_bursts_s"] == 0:
            s += ("The foam build never throttled, and peaked "
                  f"{st['peak_tj_bursts_C'] - fm['peak_tj_bursts_C']:.0f} degrees cooler. ")
        else:
            s += f"The foam build throttled for just {fm['throttled_bursts_s']:.0f}. "
        reach = max(1, round(sim.melt_reach_mm(self.wax, self.bursts[0].work))) if self.bursts else 2
        s += ("So: match the melting point to your hardware, keep all the wax within about "
              f"{reach} millimeters of metal, and expect it to buy minutes, not extra cooling. "
              "The next step is a bench test.")
        return s

    def narration(self) -> dict[str, str]:
        texts = {"intro": self.intro_say}
        texts.update({f"cue{k}": c.say for k, c in enumerate(self.cues)})
        texts["outro"] = self.outro_say
        return texts

    # ------------------------------------------------------------------ layout
    def _header(self) -> None:
        f, d, w = self.fig, self.device, self.wax
        f.text(0.025, 0.952, "Soy-wax thermal buffer for a compact AI chip", fontsize=27,
               fontweight="bold", color=TEXT, va="center")
        lo, hi = w.t_melt - 2 * w.melt_sigma, w.t_melt + 2 * w.melt_sigma
        wax_g = self.R["wax"].build.wax_mass * 1e3
        f.text(0.025, 0.914, f"Simulation prototype  ·  {d.p_full:.0f} W boost / {d.p_idle:.0f} W idle  ·  "
                             f"small fan heatsink  ·  {d.t_amb:.0f} °C air  ·  {wax_g:.0f} g soy wax melting at "
                             f"{lo:.0f}–{hi:.0f} °C", fontsize=15, color=MUTED, va="center")
        self.clock_txt = f.text(0.975, 0.952, "", fontsize=26, fontweight="bold", color=TEXT, ha="right", va="center")
        self.speed_txt = f.text(0.975, 0.914, "", fontsize=15, color=MUTED, ha="right", va="center")

    def _card(self, r, k: int) -> dict:
        f = self.fig
        cx, cw, ch = 0.025, 0.405, 0.245
        top = 0.885 - k * (ch + 0.0175)
        cy = top - ch
        bg = f.add_axes([cx, cy, cw, ch])
        bg.set_facecolor(PANEL)
        bg.set_xticks([])
        bg.set_yticks([])
        for s in bg.spines.values():
            s.set_color(GRID)
        key = r.build.key
        ax = f.add_axes([cx + 0.005, cy + 0.01, 0.16, ch - 0.02])
        n = r.layer_melt.shape[1] if r.layer_melt is not None else 0
        st = draw_stack(ax, key, n)

        lx, bx, bw = cx + 0.175, cx + 0.245, 0.075
        f.text(lx, top - 0.030, f"{LETTERS[key]}   {r.build.label}", fontsize=18, fontweight="bold",
               color=COLORS[key], va="center")
        f.text(lx, top - 0.068, "Junction temperature", fontsize=13, color=MUTED, va="center")
        tj = f.text(lx, top - 0.106, "", fontsize=30, fontweight="bold", color=TEXT, va="center")
        badge = f.text(cx + 0.300, top - 0.104, "", fontsize=13, fontweight="bold", va="center",
                       bbox=dict(boxstyle="round,pad=0.35", fc=GRID, ec="none"))
        rows = {}
        for name, dy in (("Clock", 0.152), ("Wax melted", 0.188), ("Throttled for", 0.224)):
            f.text(lx, top - dy, name, fontsize=14, color=MUTED, va="center")
            rows[name] = top - dy
        f.add_artist(Rectangle((bx, rows["Clock"] - 0.007), bw, 0.014, transform=f.transFigure, fc=GRID, ec="none"))
        clock_bar = f.add_artist(Rectangle((bx, rows["Clock"] - 0.007), bw, 0.014, transform=f.transFigure,
                                           fc=GREEN, ec="none"))
        clock_txt = f.text(bx + bw + 0.006, rows["Clock"], "", fontsize=14, color=TEXT, va="center")
        melt_bar = melt_txt = None
        if r.layer_melt is not None:
            f.add_artist(Rectangle((bx, rows["Wax melted"] - 0.007), bw, 0.014, transform=f.transFigure,
                                   fc=GRID, ec="none"))
            melt_bar = f.add_artist(Rectangle((bx, rows["Wax melted"] - 0.007), 0, 0.014, transform=f.transFigure,
                                              fc=WAX_LIQUID, ec="none"))
            melt_txt = f.text(bx + bw + 0.006, rows["Wax melted"], "", fontsize=14, color=TEXT, va="center")
        else:
            f.text(bx, rows["Wax melted"], "no wax", fontsize=14, color=MUTED, va="center")
        thr_txt = f.text(bx, rows["Throttled for"], "", fontsize=14, color=TEXT, va="center")
        return dict(r=r, chip=st["chip"], hs=st["hs"], layers=st["layers"], tj=tj, badge=badge,
                    clock_bar=clock_bar, clock_txt=clock_txt, melt_bar=melt_bar, melt_txt=melt_txt,
                    thr_txt=thr_txt, bw=bw)

    def _plots(self) -> None:
        f, d = self.fig, self.device
        x0, w = 0.47, 0.505
        self.ax_p = f.add_axes([x0, 0.715, w, 0.150])
        self.ax_t = f.add_axes([x0, 0.405, w, 0.290])
        self.ax_m = f.add_axes([x0, 0.145, w, 0.235])
        t_max = self.t_end / 60
        for ax in (self.ax_p, self.ax_t, self.ax_m):
            ax.set_facecolor(PANEL)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=12)
            ax.grid(color=GRID, lw=0.8, alpha=0.7)
            ax.set_xlim(0, t_max)
            for j in self.jobs:
                ax.axvspan(j.arrival / 60, (j.arrival + j.work) / 60, color="white", alpha=0.045, lw=0)
        for j in self.jobs:
            name = j.name if j.kind == "burst" else f"{j.work / 60:.0f}-min job"
            self.ax_p.text((j.arrival + j.work / 2) / 60, d.p_full * 1.13, name, ha="center", va="center",
                           fontsize=12, color=MUTED)
        self.ax_p.set_ylim(0, d.p_full * 1.25)
        self.ax_p.set_ylabel("Chip power (W)", color=TEXT, fontsize=14)
        self.ax_t.set_ylim(30, 95)
        self.ax_t.set_ylabel("Junction temperature (°C)", color=TEXT, fontsize=14)
        self.ax_t.axhline(d.t_limit, color=RED, ls="--", lw=1.4)
        self.ax_t.text(t_max - 0.3, d.t_limit + 1.6, f"throttle limit {d.t_limit:.0f} °C", ha="right",
                       va="center", fontsize=12, color=RED)
        self.ax_m.set_ylim(-3, 108)
        self.ax_m.set_ylabel("Wax melted (%)", color=TEXT, fontsize=14)
        self.ax_m.set_xlabel("Device time (min)", color=TEXT, fontsize=14)
        self.ax_m.text(0.3, 98, "pack color:  cream = solid wax,  amber = melted", fontsize=12, color=MUTED, va="center")
        self.ax_p.tick_params(labelbottom=False)
        self.ax_t.tick_params(labelbottom=False)
        handles = [Line2D([], [], color=COLORS[r.build.key], lw=3, label=f"{LETTERS[r.build.key]}  {r.build.label}")
                   for r in self.shown]
        f.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0 - 0.005, 0.866), ncol=3, frameon=False,
                 fontsize=13, labelcolor=TEXT, handlelength=1.6, columnspacing=1.6)
        self.lines = []
        for r in self.shown:
            c = COLORS[r.build.key]
            entry = dict(r=r, p=self.ax_p.plot([], [], color=c, lw=2)[0],
                         t=self.ax_t.plot([], [], color=c, lw=2.4)[0],
                         t_dot=self.ax_t.plot([], [], "o", color=c, ms=7)[0], m=None, m_dot=None)
            if r.layer_melt is not None:
                entry["m"] = self.ax_m.plot([], [], color=c, lw=2.4)[0]
                entry["m_dot"] = self.ax_m.plot([], [], "o", color=c, ms=7)[0]
            self.lines.append(entry)
        self.now = [ax.axvline(0, color="white", lw=1, alpha=0.35) for ax in (self.ax_p, self.ax_t, self.ax_m)]
        self.minutes = self.shown[0].t / 60

    def _caption_bar(self) -> None:
        f = self.fig
        f.add_artist(Rectangle((0.025, 0.014), 0.95, 0.078, transform=f.transFigure, fc=PANEL, ec=GRID, lw=1))
        self.caption = f.text(0.5, 0.053, "", fontsize=21, color=TEXT, ha="center", va="center", linespacing=1.35)
        f.add_artist(Rectangle((0, 0), 1, 0.006, transform=f.transFigure, fc=GRID, ec="none"))
        self.progress = f.add_artist(Rectangle((0, 0), 0, 0.006, transform=f.transFigure, fc=COLORS["foam"], ec="none"))

    # ------------------------------------------------------------------ title and results cards
    def _overlay(self, n_groups: int = 0) -> tuple:
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zorder(50)
        ov = dict(axes=[ax], base=[], groups=[[] for _ in range(n_groups)])
        ov["base"].append((ax.add_patch(Rectangle((0, 0), 1, 1, fc=BG, ec="none")), 1.0))
        return ax, ov

    def _intro(self) -> dict:
        ax, ov = self._overlay(n_groups=len(self.intro_keys) + 1)

        def add(g: int, artist, base: float = 1.0):
            ov["groups"][g].append((artist, base))
            return artist

        wax_g = self.R["wax"].build.wax_mass * 1e3
        tb, tc = self.R["wax"].pack["thickness_mm"], self.R["foam"].pack["thickness_mm"]
        side = math.sqrt(self.R["wax"].build.area) * 1e3
        add(0, ax.text(0.07, 0.80, "Can soy wax smooth heat spikes\nin compact AI hardware?", fontsize=50,
                       fontweight="bold", color=TEXT, va="top", linespacing=1.15))
        add(1, ax.text(0.07, 0.535, "A simulation prototype: same chip, same fan, same AI workload, three builds.",
                       fontsize=23, color=MUTED, va="center"))
        rows = [("stock", "Stock heatsink and fan"),
                ("wax", f"+ {wax_g:.0f} g soy wax pack ({side:.0f} × {side:.0f} × {tb:.0f} mm) on the heatsink"),
                ("foam", f"+ {wax_g:.0f} g soy wax cast into aluminum foam ({side:.0f} × {side:.0f} × {tc:.0f} mm)")]
        for k, (key, label) in enumerate(rows):
            y = 0.43 - k * 0.07
            add(2 + k, ax.add_patch(Rectangle((0.07, y - 0.017), 0.009, 0.034, fc=COLORS[key], ec="none")))
            add(2 + k, ax.text(0.088, y, f"{LETTERS[key]}    {label}", fontsize=24, color=TEXT, va="center"))
        nb = len(self.bursts)
        wl = "Workload:"
        if nb:
            wl += f" {NUM_WORDS.get(nb, nb)} {self.bursts[0].work:.0f}-second AI jobs"
        if self.sustained:
            m = round(self.sustained.work / 60)
            wl += f", then {article(m)} {m}-minute job"
        wl += f"  ·  {self.t_end / 60:.0f} minutes of device time, sped up"
        add(5, ax.text(0.07, 0.17, wl, fontsize=19, color=MUTED, va="center"))
        add(1, ax.text(0.07, 0.085, "Physics model, not a lab measurement. Assumptions and sources are in README.md.",
                       fontsize=15, color=MUTED, va="center", style="italic"))
        sax = self.fig.add_axes([0.685, 0.33, 0.27, 0.40])  # illustration of build C
        sax.set_zorder(51)
        ov["axes"].append(sax)
        st = draw_stack(sax, "foam", 12, size=14)
        st["chip"].set_facecolor(TEMP_CMAP(TEMP_NORM(80)))
        for p in st["hs"]:
            p.set_facecolor(TEMP_CMAP(TEMP_NORM(54)))
        for i, p in enumerate(st["layers"]):
            p.set_facecolor(wax_color(max(0.0, 1 - i / 5)))
        for a in st["artists"]:
            add(4, a, HATCH_ALPHA if a is st["hatch"] else 1.0)
        add(4, sax.text(5.0, -0.9, "Build C, side view", ha="center", va="center", fontsize=14, color=MUTED))
        return ov

    def _outro(self) -> dict:
        ax, ov = self._overlay()

        def add(artist):
            ov["base"].append((artist, 1.0))
            return artist

        S, d, w = self.S, self.device, self.wax
        add(ax.text(0.06, 0.905, "What the simulation shows", fontsize=36, fontweight="bold", color=TEXT, va="center"))
        add(ax.text(0.06, 0.855, "Same chip, fan and workload. Only the buffer changes. "
                                 "D is a control: the same mass of plain aluminum.", fontsize=17, color=MUTED, va="center"))
        long_job = f"{round(self.sustained.work / 60)}-min job" if self.sustained else "long job"
        cols = [(0.06, "Build", "left"), (0.47, "Peak junction\n(3 short jobs)", "center"),
                (0.60, "Rise per job", "center"), (0.72, "Throttled\n(3 short jobs)", "center"),
                (0.86, f"Full speed in\n{long_job}", "center")]
        for x, name, ha in cols:
            add(ax.text(x, 0.775, name, fontsize=15, color=MUTED, ha=ha, va="center", linespacing=1.2))
        add(ax.add_patch(Rectangle((0.05, 0.709), 0.90, 0.002, fc=GRID, ec="none")))
        for k, key in enumerate(("stock", "wax", "foam", "alu")):
            s = S[key]
            yy = 0.672 - k * 0.058
            if key == "foam":
                add(ax.add_patch(Rectangle((0.05, yy - 0.026), 0.90, 0.052, fc=PANEL, ec=COLORS["foam"], lw=1.2)))
            vals = [f"{s['peak_tj_bursts_C']:.1f} °C", f"{s['rise_per_burst_K']:.0f} K",
                    f"{s['throttled_bursts_s']:.0f} s", sim.mmss(s.get("sustained_full_speed_s", math.nan))]
            add(ax.text(0.06, yy, f"{LETTERS[key]}   {s['build']}", fontsize=20, color=COLORS[key],
                        fontweight="bold", va="center"))
            for (x, _, ha), v in zip(cols[1:], vals):
                add(ax.text(x, yy, v, fontsize=20, color=TEXT, ha=ha, va="center"))
        foam, stock = S["foam"], S["stock"]
        lo, hi = w.t_melt - 2 * w.melt_sigma, w.t_melt + 2 * w.melt_sigma
        hs_idle = d.t_amb + d.p_idle * d.r_ha
        hs_thr = d.t_limit - d.p_full * d.r_jc
        thick = self.R["wax"].pack["thickness_mm"]
        fr = self.R["foam"]
        i1 = self.jobs.index(self.bursts[0])
        t_solid = first_time(fr.melt < 0.03, fr.t, fr.job_end[i1], self.t_end)
        refreeze = (t_solid - fr.job_end[i1]) / 60 if t_solid is not None else math.nan
        bullets = [
            f"Match the melting point to the hardware. This wax melts at {lo:.0f}–{hi:.0f} °C, between the "
            f"{hs_idle:.0f} °C idle heatsink and the {hs_thr:.0f} °C where throttling starts.",
            f"Pure wax conducts heat poorly (~{w.k_solid:.2f} W/m·K). The {thick:.0f} mm slab did about as well as "
            "plain aluminum. Wax in metal foam or between fins makes it work.",
            f"It buys time, not capacity: {sim.mmss(foam['sustained_full_speed_s'])} at full speed instead of "
            f"{sim.mmss(stock['sustained_full_speed_s'])}, then every build throttles to the same power.",
            f"It has to re-solidify: about {refreeze:.0f} min of idle after each job, with the chip idling a few "
            "degrees warmer meanwhile.",
        ]
        for k, b in enumerate(bullets):
            add(ax.text(0.06, 0.39 - k * 0.062, "•  " + b, fontsize=17, color=TEXT, va="center"))
        add(ax.text(0.06, 0.07, "Model: lumped chip + heatsink, 1-D enthalpy model of the wax pack, DVFS governor. "
                                f"Soy wax: {w.latent / 1e3:.0f} J/g, ~{w.k_solid:.2f} W/m·K. Wax in aluminum foam: "
                                f"{fr.build.k_pack:.0f} W/m·K. Validate on a bench rig before relying on these numbers.",
                    fontsize=13, color=MUTED, va="center", style="italic"))
        return ov

    # ------------------------------------------------------------------ per-frame updates
    def _apply_overlay(self, ov: dict, alpha: float, t: float | None = None) -> None:
        for ax in ov["axes"]:
            ax.set_visible(alpha > 0)
        if alpha <= 0:
            return
        for art, base in ov["base"]:
            art.set_alpha(base * alpha)
        for g, items in enumerate(ov["groups"]):
            r = 1.0 if t is None else min(1.0, max(0.0, (t - self.intro_reveal[g]) / REVEAL_S))
            for art, base in items:
                art.set_alpha(base * alpha * r)

    def render(self, fr: Frame, still: bool = False) -> None:
        self._apply_overlay(self.intro, fr.alpha if fr.overlay == "intro" else 0.0, fr.intro_t)
        self._apply_overlay(self.outro, fr.alpha if fr.overlay == "outro" else 0.0)
        t_sim = fr.t
        i = min(int(round(t_sim / self.dt_rec)), len(self.minutes) - 1)
        d = self.device
        self.clock_txt.set_text(f"t = {sim.mmss(t_sim)}")
        self.speed_txt.set_text(f"playback {fr.speed:.0f}×" if t_sim > 0 else "")
        x = self.minutes[: i + 1]
        for e in self.lines:
            r = e["r"]
            e["p"].set_data(x, r.power[: i + 1])
            e["t"].set_data(x, r.t_j[: i + 1])
            e["t_dot"].set_data([x[-1]], [r.t_j[i]])
            if e["m"] is not None:
                e["m"].set_data(x, 100 * r.melt[: i + 1])
                e["m_dot"].set_data([x[-1]], [100 * r.melt[i]])
        for ln in self.now:
            ln.set_xdata([x[-1], x[-1]])
        for c in self.cards:
            r = c["r"]
            tj, ths, busy, s = r.t_j[i], r.t_hs[i], bool(r.busy[i]), r.clock[i]
            throttled = busy and s < sim.THROTTLED_BELOW
            c["chip"].set_facecolor(TEMP_CMAP(TEMP_NORM(tj)))
            hs_col = TEMP_CMAP(TEMP_NORM(ths))
            for p in c["hs"]:
                p.set_facecolor(hs_col)
            if c["layers"]:
                for p, fm in zip(c["layers"], r.layer_melt[i]):
                    p.set_facecolor(wax_color(fm))
                c["melt_bar"].set_width(c["bw"] * r.melt[i])
                c["melt_txt"].set_text(f"{100 * r.melt[i]:.0f} %")
            c["tj"].set_text(f"{tj:.1f} °C")
            c["tj"].set_color(RED if tj >= d.t_limit - 0.5 else TEXT)
            if throttled:
                state, fc, tc = "THROTTLED", RED, BG
            elif busy:
                state, fc, tc = "FULL SPEED", GREEN, BG
            else:
                state, fc, tc = "IDLE", GRID, TEXT
            c["badge"].set_text(state)
            c["badge"].set_color(tc)
            c["badge"].get_bbox_patch().set_facecolor(fc)
            if busy:
                c["clock_bar"].set_width(c["bw"] * s)
                c["clock_bar"].set_facecolor(RED if throttled else GREEN)
                c["clock_txt"].set_text(f"{100 * s:.0f} %  ·  {r.power[i]:.0f} W")
            else:
                c["clock_bar"].set_width(0)
                c["clock_txt"].set_text(f"idle  ·  {r.power[i]:.0f} W")
            c["thr_txt"].set_text(f"{sim.mmss(self.thr_cum[r.build.key][i])}  (m:ss)")
        if fr.cue >= 0:
            self.caption.set_text(self.cue_text[fr.cue])
            self.caption.set_alpha(1.0 if still else min(1.0, (fr.cue_age + 1) / 8))
        else:
            self.caption.set_text("")
        self.progress.set_width(min(1.0, t_sim / self.t_end))


# ---------------------------------------------------------------------------- timeline
def speed_target(jobs: list[sim.Job]):
    """Preferred playback speed (simulated s per video s): slow while jobs run, fast through idle time."""
    segs = []
    first = True
    for j in jobs:
        if j.kind == "burst":
            segs.append((j.arrival - 15, j.arrival + j.work + (65 if first else 45), 8.0 if first else 18.0))
            first = False
        else:
            segs.append((j.arrival - 15, j.arrival + 330, 20.0))
    t_first = min((j.arrival for j in jobs), default=0.0)

    def target(t: float) -> float:
        for a, b, v in segs:
            if a <= t < b:
                return v
        return 20.0 if t < t_first else 60.0

    return target


def playback(demo: Demo, slow: list[float], fps: int) -> tuple[list[tuple[float, float]], list[int]]:
    """Frames of the simulation playback. Cue k's stretch runs `slow[k]` times slower than preferred.
    Returns [(sim time, speed)] and the first frame of every cue."""
    target = speed_target(demo.jobs)
    starts = [c.t for c in demo.cues]
    ease = 1 - math.exp(-1 / (fps * 0.5))
    out: list[tuple[float, float]] = []
    first = [0] * len(starts)
    t, k = 0.0, 0
    v = target(0.0) / slow[0]
    while t < demo.t_end:
        while k + 1 < len(starts) and t >= starts[k + 1]:
            k += 1
            first[k] = len(out)
        out.append((t, v))
        v += (target(t) / slow[k] - v) * ease
        t += v / fps
    return out, first


def fit_playback(demo: Demo, fps: int, need: list[float]) -> tuple[list[tuple[float, float]], list[int]]:
    """Slow down each cue's stretch until its narration (need[k] seconds) fits before the next cue."""
    slow = [1.0] * len(need)
    for _ in range(100):
        pb, first = playback(demo, slow, fps)
        bounds = first + [len(pb)]
        done = True
        for k, n in enumerate(need):
            avail = max(1, bounds[k + 1] - bounds[k]) / fps
            if avail < n:
                slow[k] *= min(3.0, max(1.01, n / avail))
                done = False
        if done:
            return pb, first
    print("warning: narration timing did not fully converge")
    return pb, first


def build_plan(demo: Demo, fps: int, clips: dict[str, Clip]) -> tuple[list[Frame], list[tuple[float, str]]]:
    """Whole video: intro card, playback, end hold, results card.
    Returns the frames and the narration schedule [(video time s, clip key)]."""
    K = len(demo.cues)
    fade = int(round(0.7 * fps))
    intro = clips["intro"]
    demo.intro_reveal = [0.0] + [INTRO_LEAD + m - 0.1 for m in find_marks(intro, demo.intro_keys)]
    v0 = speed_target(demo.jobs)(0.0)
    frames: list[Frame] = []
    sched: list[tuple[float, str]] = [(INTRO_LEAD, "intro")]
    n_hold = int(round(max(5.0, INTRO_LEAD + intro.duration + 1.0) * fps))
    for n in range(n_hold + fade):
        a = 1.0 if n < n_hold else 1 - (n - n_hold + 1) / fade
        frames.append(Frame(0.0, v0, "intro", a, -1, 0, n / fps))
    frames += [Frame(0.0, v0)] * int(round(0.8 * fps))

    pb, first = fit_playback(demo, fps, [clips[f"cue{k}"].duration + GAP for k in range(K)])
    base = len(frames)
    sched += [((base + n0) / fps, f"cue{k}") for k, n0 in enumerate(first)]
    k = 0
    for n, (t, v) in enumerate(pb):
        while k + 1 < K and n >= first[k + 1]:
            k += 1
        frames.append(Frame(t, v, None, 0.0, k, n - first[k]))

    v_end = pb[-1][1]
    t_last = sched[-1][0] + clips[f"cue{K - 1}"].duration
    n_end = max(int(round(1.2 * fps)), math.ceil((t_last + 0.8) * fps) - len(frames))
    age = len(frames) - base - first[-1]
    for j in range(n_end + fade):
        a = 0.0 if j < n_end else (j - n_end + 1) / fade
        frames.append(Frame(demo.t_end, v_end, "outro" if a > 0 else None, a, K - 1, age + j))
    sched.append((len(frames) / fps + 0.3, "outro"))
    n_out = int(round(max(10.0, 0.3 + clips["outro"].duration + 2.5) * fps))
    frames += [Frame(demo.t_end, v_end, "outro", 1.0, K - 1, age + n_end + fade)] * n_out
    return frames, sched


# ---------------------------------------------------------------------------- voice
def _clip_files(cache: Path, voice: str, rate: str, text: str) -> tuple[Path, Path]:
    h = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    return cache / f"{h}.mp3", cache / f"{h}.json"


async def _synth_all(todo: list[tuple[str, Path, Path]], voice: str, rate: str) -> None:
    import edge_tts

    sem = asyncio.Semaphore(4)

    async def one(text: str, mp3: Path, meta: Path) -> None:
        async with sem:
            err: Exception | None = None
            for attempt in range(5):
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
                    audio, words = bytearray(), []
                    async for ch in comm.stream():
                        if ch["type"] == "audio":
                            audio += ch["data"]
                        elif ch["type"] == "WordBoundary":
                            words.append([ch["offset"] / 1e7, ch["text"]])
                    if not audio:
                        raise RuntimeError("no audio received")
                    meta.write_text(json.dumps(words))
                    part = mp3.with_suffix(".part")
                    part.write_bytes(bytes(audio))
                    part.replace(mp3)
                    return
                except Exception as e:  # network hiccups or service throttling: retry
                    err = e
                    await asyncio.sleep(1.5 * (attempt + 1))
            raise RuntimeError(f"edge-tts failed for voice {voice!r} after 5 tries: {err}")

    await asyncio.gather(*(one(*job) for job in todo))


def decode_audio(path: Path) -> np.ndarray:
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def trim_silence(a: np.ndarray, thresh_db: float = -50.0) -> tuple[float, np.ndarray]:
    idx = np.flatnonzero(np.abs(a) > 10 ** (thresh_db / 20))
    if idx.size == 0:
        return 0.0, a[:0]
    i0 = max(0, int(idx[0]) - int(0.03 * SR))
    i1 = min(len(a), int(idx[-1]) + int(0.12 * SR))
    return i0 / SR, a[i0:i1]


def synthesize(texts: dict[str, str], cache: Path, voice: str, rate: str) -> dict[str, Clip]:
    cache.mkdir(parents=True, exist_ok=True)
    todo = []
    for text in dict.fromkeys(texts.values()):
        mp3, meta = _clip_files(cache, voice, rate, text)
        if not (mp3.exists() and meta.exists()):
            todo.append((text, mp3, meta))
    if todo:
        t0 = time.time()
        print(f"synthesizing {len(todo)} voice clips with {voice} ...", flush=True)
        asyncio.run(_synth_all(todo, voice, rate))
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    clips = {}
    for key, text in texts.items():
        mp3, meta = _clip_files(cache, voice, rate, text)
        lead, audio = trim_silence(decode_audio(mp3))
        words = tuple((max(0.0, s - lead), w) for s, w in json.loads(meta.read_text()))
        clips[key] = Clip(key, text, len(audio) / SR, audio, words)
    return clips


def silent_clips(texts: dict[str, str]) -> dict[str, Clip]:
    return {k: Clip(k, v, len(v.split()) / WORDS_PER_S + 0.4) for k, v in texts.items()}


def find_marks(clip: Clip, keys: list[str]) -> list[float]:
    """Time (s into the clip) at which each key word is spoken, in order."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    if clip.words:
        ws = [(s, norm(w)) for s, w in clip.words]
        out, pos = [], 0
        for key in keys:
            j = next((n for n in range(pos, len(ws)) if ws[n][1] == key), None)
            if j is None:
                break
            out.append(ws[j][0])
            pos = j + 1
        if len(out) == len(keys):
            return out
    low, out, pos = clip.text.lower(), [], 0  # fallback: proportional to character position
    for key in keys:
        j = low.find(key, pos)
        j = pos if j < 0 else j
        out.append(clip.duration * j / max(1, len(low)))
        pos = j + 1
    return out


def mix_narration(sched: list[tuple[float, str]], clips: dict[str, Clip], total_s: float, path: Path) -> tuple[float, float]:
    n = int(round(total_s * SR))
    buf = np.zeros(n)
    for start, key in sched:
        a = clips[key].audio
        i = int(round(start * SR))
        seg = a[: max(0, n - i)]
        buf[i:i + len(seg)] += seg
    loud = buf[np.abs(buf) > 10 ** (-40 / 20)]
    rms = math.sqrt(float(np.mean(loud ** 2)))
    peak = float(np.abs(buf).max())
    gain = min(10 ** (-19 / 20) / rms, 10 ** (-1.5 / 20) / peak)
    pcm = np.clip(np.round(buf * gain * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:  # true stereo: an encoder upmix would drop the level by 3 dB
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm, 2).tobytes())
    return 20 * math.log10(rms * gain), 20 * math.log10(peak * gain)


# ---------------------------------------------------------------------------- rendering
def make_demo(scenario: argparse.Namespace) -> Demo:
    device, wax, builds, jobs = sim.build_scenario(scenario)
    results = sim.run_all(device, wax, builds, jobs)
    summaries = [sim.summarize(r) for r in results]
    return Demo(results, summaries, device, wax, jobs)


def render_chunk(spec: dict) -> tuple[str, float]:
    """Worker process: rebuild the (deterministic) demo and plan, render frames [start, stop)."""
    t0 = time.time()
    setup_fonts()
    demo = make_demo(argparse.Namespace(**spec["scenario"]))
    frames, _ = build_plan(demo, spec["fps"], spec["clips"])
    if len(frames) != spec["n_frames"]:
        raise RuntimeError(f"worker plan has {len(frames)} frames, expected {spec['n_frames']}")
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
    writer = FFMpegWriter(fps=spec["fps"], codec="libx264", bitrate=-1,
                          extra_args=["-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv444p", "-threads", "3"])
    with writer.saving(demo.fig, spec["path"], dpi=100):
        for fr in frames[spec["start"]:spec["stop"]]:
            demo.render(fr)
            writer.grab_frame()
    plt.close(demo.fig)
    return spec["path"], time.time() - t0


def render_video(scenario: dict, clips: dict[str, Clip], n_frames: int, fps: int, workers: int,
                 tmp: Path) -> list[Path]:
    light = {k: replace(c, audio=None) for k, c in clips.items()}
    n_chunks = max(1, min(workers, n_frames // 60))
    bounds = np.linspace(0, n_frames, n_chunks + 1).round().astype(int)
    specs = [dict(scenario=scenario, clips=light, fps=fps, start=int(a), stop=int(b), n_frames=n_frames,
                  path=str(tmp / f"seg_{k:02d}.mp4")) for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    t0 = time.time()
    print(f"rendering {n_frames} frames in {n_chunks} parallel chunks ...", flush=True)
    with ProcessPoolExecutor(max_workers=n_chunks) as ex:
        futures = [ex.submit(render_chunk, s) for s in specs]
        for done, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  chunk {done}/{n_chunks} done ({time.time() - t0:.0f} s)", flush=True)
    return [Path(s["path"]) for s in specs]


def encode_final(segments: list[Path], wav: Path | None, out: Path, tmp: Path) -> None:
    lst = tmp / "segments.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments))
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if wav is not None:
        cmd += ["-i", str(wav)]
    cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    if wav is not None:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", str(SR)]
    cmd += ["-movflags", "+faststart", "-metadata", f"title={TITLE}", str(out)]
    subprocess.run(cmd, check=True)


def mmss_frac(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main() -> None:
    sp = argparse.ArgumentParser(add_help=False)
    sim.add_scenario_args(sp)
    ap = argparse.ArgumentParser(description=__doc__, parents=[sp], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/soywax_demo.mp4", help="output MP4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills", action="store_true", help="write PNG stills next to --out instead of a video")
    ap.add_argument("--no-voice", action="store_true", help="skip the voiceover (captions only, works offline)")
    ap.add_argument("--voice", default=VOICE, help=f"edge-tts voice (default {VOICE})")
    ap.add_argument("--rate", default=RATE, help="speaking-rate change such as +10%% (default +4%%)")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)),
                    help="parallel render processes")
    args = ap.parse_args()
    scenario = vars(sp.parse_known_args()[0])

    setup_fonts()
    demo = make_demo(argparse.Namespace(**scenario))
    texts = demo.narration()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clips = silent_clips(texts) if args.no_voice else synthesize(texts, out.parent / "voice", args.voice, args.rate)
    frames, sched = build_plan(demo, args.fps, clips)
    total = len(frames) / args.fps

    lines = []
    for start, key in sched:
        c = clips[key]
        lines.append(f"[{mmss_frac(start)} - {mmss_frac(start + c.duration)}] {c.text}")
    overlap = max((sched[i][0] + clips[sched[i][1]].duration - sched[i + 1][0] for i in range(len(sched) - 1)),
                  default=0.0)
    script = out.with_name(out.stem + "_narration.txt")
    script.write_text(f"{TITLE}\nvoice: {'none' if args.no_voice else args.voice}, video length {mmss_frac(total)}\n\n"
                      + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"timeline: {len(frames)} frames = {total:.1f} s; {len(sched)} narration lines; "
          f"worst overlap between lines {overlap:+.2f} s (negative = gap)")
    print(f"wrote {script}")

    if args.stills:
        intro_full = [n for n, f in enumerate(frames) if f.overlay == "intro" and f.alpha == 1.0]
        picks = {"intro_reveal": intro_full[len(intro_full) // 3], "intro": intro_full[-1]}
        for name, ts in (("job1", 140.0), ("idle", 300.0), ("job2", 560.0), ("job3", 975.0),
                         ("sustained", 1560.0), ("end", demo.t_end)):
            picks[name] = next(n for n, f in enumerate(frames) if f.t >= ts and f.overlay is None)
        picks["outro"] = len(frames) - 1
        for name, n in picks.items():
            demo.render(frames[n], still=True)
            path = out.with_name(f"still_{name}.png")
            demo.fig.savefig(path, dpi=100)
            print("wrote", path)
        return

    tmp = out.parent / "_render"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.time()
    segments = render_video(scenario, clips, len(frames), args.fps, args.workers, tmp)
    wav = None
    if not args.no_voice:
        wav = tmp / "narration.wav"
        rms_db, peak_db = mix_narration(sched, clips, total, wav)
        print(f"narration mixed: speech RMS {rms_db:.1f} dBFS, peak {peak_db:.1f} dBFS")
    print("encoding final MP4 ...", flush=True)
    encode_final(segments, wav, out, tmp)
    shutil.rmtree(tmp)
    print(f"wrote {out}: {total:.1f} s, {len(frames)} frames, {time.time() - t0:.0f} s to render")


if __name__ == "__main__":
    main()
