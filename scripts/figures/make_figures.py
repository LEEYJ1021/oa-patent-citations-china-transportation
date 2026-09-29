#!/usr/bin/env python3
"""
Journal-style, black-and-white figures for the revised manuscript.
Fig. 2  sample flow            Fig. 6  H1 forest plot
Fig. 7  H2a visibility         Fig. 8  H2b period rate ratios
Fig. 11 colour forest + contrasts (H3)
Fig. N1 leave-one-institution-out (needs C3_leave_one_institution_out.csv)
Fig. N2 Lens vs Unpaywall colour agreement

Numbers are copied from the analysis ledger (FINAL_REPORT / STEP6_REPORT).
Run:  python3 make_figures.py [outdir] [path/to/C3_leave_one_institution_out.csv]
Output: PNG (600 dpi) + PDF (vector, editable text) per figure.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

OUT = sys.argv[1] if len(sys.argv) > 1 else "./figures"
C3 = sys.argv[2] if len(sys.argv) > 2 else "./C3_leave_one_institution_out.csv"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "figure.dpi": 150, "axes.unicode_minus": True,
})
K = "black"
G1, G2 = "0.55", "0.85"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("saved", name)


def panel(ax, s, dx=-0.0, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def logticks(ax, ticks, axis="x"):
    lab = [("%g" % t) for t in ticks]
    if axis == "x":
        ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(lab)); ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_yscale("log"); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter(lab)); ax.yaxis.set_minor_locator(NullLocator())


def fmt(rr, lo, hi):
    return f"{rr:.2f} ({lo:.2f}\u2013{hi:.2f})"


# ─────────────────────────── Fig. 2: sample flow ───────────────────────────
def fig2():
    fig, ax = plt.subplots(figsize=(6.8, 6.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")

    def box(cx, cy, w, h, text, lw=0.9, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", fc="white", ec=K, lw=lw, ls=ls))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8, fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=K, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=8))

    cx, w, h = 30, 38, 10
    ys = [106, 88, 70, 52, 34]
    chain = ["Raw Lens.org export\n13,350 rows", "12,350 rows", "12,348 records",
             "12,302 records\n(10,114 unique Lens IDs)",
             "Main analysis sample, 2003\u20132024\n12,020 records (paper\u2013institution pairs)"]
    excl = ["Excluded: empty padding rows\n(n = 1,000)", "Excluded: missing patent-citation count\n(n = 2)",
            "Excluded: missing publication year (n = 42);\npublication year 2026 (n = 4)",
            "Excluded: published before 2003 (n = 26);\npublished in 2025 (n = 256)"]
    for i, (y, t) in enumerate(zip(ys, chain)):
        box(cx, y, w if i < 4 else 46, h if i < 4 else 12, t, lw=1.3 if i == 4 else 0.9, bold=(i == 4))
    for i in range(4):
        y_top = ys[i] - (h / 2 if i < 4 else 6); y_bot = ys[i + 1] + (h / 2 if i + 1 < 4 else 6)
        arrow(cx, y_top, cx, y_bot)
        ym = (y_top + y_bot) / 2
        ax.plot([cx, 54], [ym, ym], color=K, lw=0.8)
        box(77, ym, 46, 10, excl[i])
    # sub-samples
    yb = 9
    xs = [16, 50, 84]
    subs = ["One row per paper\n(Lens ID)\nn = 9,903", "Colour models\n('other' colour excluded)\nn = 11,955",
            "H2b sub-sample\npublished 2003\u20132019\nn = 6,640"]
    ax.plot([cx, cx], [ys[4] - 6, 22], color=K, lw=0.8)
    ax.plot([xs[0], xs[2]], [22, 22], color=K, lw=0.8)
    for x, t in zip(xs, subs):
        arrow(x, 22, x, yb + 7)
        box(x, yb, 30, 14, t, ls=(0, (4, 2)))
    ax.text(0, 118, "All records have at least one citing patent by construction.", fontsize=7.5, style="italic", va="top")
    save(fig, "Fig2_sample_flow")


# ─────────────────────────── Fig. 6: H1 forest ───────────────────────────
def fig6():
    rows = [("H", "Zero-truncated Poisson, all records"),
            ("R", "Unadjusted", 1.524, 1.361, 1.706, 12020, 0),
            ("R", "+ academic citations", 1.338, 1.211, 1.480, 12020, 1),
            ("R", "+ academic citations, publication type, field tags", 1.374, 1.249, 1.511, 12020, 1),
            ("H", "Gold/Hybrid (publisher-hosted OA) vs closed"),
            ("R", "Unadjusted", 1.289, 1.074, 1.547, 10553, 0),
            ("R", "+ academic citations", 1.258, 1.053, 1.502, 10553, 1),
            ("H", "Zero-truncated Poisson, one row per paper"),
            ("R", "Unadjusted", 1.526, 1.342, 1.734, 9903, 0),
            ("R", "+ academic citations", 1.329, 1.198, 1.474, 9903, 1),
            ("H", "Zero-truncated negative binomial (direction check)"),
            ("R", "Unadjusted", 1.670, 1.409, 1.980, 12020, 0),
            ("R", "+ academic citations", 1.416, 1.204, 1.667, 12020, 1)]
    n = len(rows)
    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_axes([0.40, 0.14, 0.33, 0.80])
    ypos = np.arange(n)[::-1]
    ax.set_ylim(-0.8, n - 0.2)
    ax.axvline(1, color=K, lw=0.7, ls=(0, (3, 2)))
    yt, yl = [], []
    tr = ax.get_yaxis_transform()
    ax.text(1.06, n - 0.35, "RR (95% CI)", transform=tr, fontsize=8, fontweight="bold", va="center")
    ax.text(1.62, n - 0.35, "N", transform=tr, fontsize=8, fontweight="bold", va="center", ha="right")
    for y, r in zip(ypos, rows):
        if r[0] == "H":
            ax.text(-1.05, y, r[1], transform=tr, fontsize=8, fontweight="bold", va="center", ha="left", bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
            continue
        _, lab, rr, lo, hi, N, adj = r
        ax.plot([lo, hi], [y, y], color=K, lw=1.1, solid_capstyle="butt")
        ax.plot([lo, lo], [y - .16, y + .16], color=K, lw=1.1); ax.plot([hi, hi], [y - .16, y + .16], color=K, lw=1.1)
        ax.plot(rr, y, "o", ms=5.5, mfc="white" if adj else K, mec=K, mew=1.1, zorder=3)
        ax.text(-0.03, y, lab, transform=tr, fontsize=8, va="center", ha="right")
        ax.text(1.06, y, fmt(rr, lo, hi), transform=tr, fontsize=8, va="center")
        ax.text(1.62, y, f"{N:,}", transform=tr, fontsize=8, va="center", ha="right")
    logticks(ax, [0.9, 1, 1.25, 1.5, 2.0]); ax.set_xlim(0.9, 2.05)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Rate ratio, OA vs closed (log scale)")
    ax.plot([], [], "o", ms=5.5, mfc=K, mec=K, label="Without academic-citation adjustment")
    ax.plot([], [], "o", ms=5.5, mfc="white", mec=K, mew=1.1, label="With academic-citation adjustment")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), frameon=False, ncol=2, handletextpad=0.3, columnspacing=1.2)
    save(fig, "Fig6_H1_forest")


# ─────────────────────────── Fig. 7: H2a visibility ───────────────────────────
def fig7():
    mu, b0, b1 = 3.59, np.log(1.192), np.log(1.146)
    a, c, e = 0.006006, -0.003878, 0.003787          # var(d) = a + 2 c d + e d^2, d = lc - mu
    tcrit = 2.0639
    lc = np.linspace(1.0, 6.0, 300); d = lc - mu
    se = np.sqrt(a + 2 * c * d + e * d ** 2); mid = b0 + b1 * d
    fig = plt.figure(figsize=(7.0, 3.2))
    ax = fig.add_axes([0.075, 0.17, 0.52, 0.74]); ax2 = fig.add_axes([0.73, 0.17, 0.25, 0.74])
    ax.fill_between(lc, np.exp(mid - tcrit * se), np.exp(mid + tcrit * se), color=G2, lw=0)
    ax.plot(lc, np.exp(mid), color=K, lw=1.4)
    ax.axhline(1, color=K, lw=0.7, ls=(0, (3, 2)))
    pts = [(1.61, 0.911, 0.615, 1.349, "10th"), (3.64, 1.200, 1.028, 1.401, "50th"), (5.39, 1.523, 1.330, 1.744, "90th")]
    for x, rr, lo, hi, lab in pts:
        ax.plot([x, x], [lo, hi], color=K, lw=1.1)
        ax.plot(x, rr, "o", ms=5.5, mfc=K, mec=K, zorder=3)
        ax.annotate(f"{lab} pct.\n{rr:.2f}", (x, hi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    logticks(ax, [0.6, 0.75, 1, 1.5, 2, 2.5], "y"); ax.set_ylim(0.6, 2.6); ax.set_xlim(1, 6)
    ax.set_xlabel("Academic citations, log(1 + citations)"); ax.set_ylabel("Rate ratio, OA vs closed (log scale)")
    ax.text(0.03, 0.97, "OA \u00d7 visibility: RR 1.15 per log-unit\n(95% CI 1.01\u20131.30; wild-bootstrap p = .012)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left")
    panel(ax, "a", dx=-0.02 - 0.075 * 0, dy=1.02)
    # panel b: terciles
    t = [("Highest third", 1.751, 1.478, 2.074), ("Middle third", 1.183, 0.940, 1.489), ("Lowest third", 1.079, 0.743, 1.569)]
    for i, (lab, rr, lo, hi) in enumerate(t):
        y = 2 - i
        ax2.plot([lo, hi], [y, y], color=K, lw=1.1)
        ax2.plot([lo, lo], [y - .12, y + .12], color=K, lw=1.1); ax2.plot([hi, hi], [y - .12, y + .12], color=K, lw=1.1)
        ax2.plot(rr, y, "s", ms=5.5, mfc=K, mec=K, zorder=3)
        near1 = np.log(rr) < 0.25            # keep the value label clear of the RR = 1 line
        ax2.text(1.05 if near1 else rr, y + 0.28, f"{rr:.2f}", ha="left" if near1 else "center", fontsize=7.5, zorder=5)
    ax2.axvline(1, color=K, lw=0.7, ls=(0, (3, 2)))
    ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels([x[0] for x in t][::-1]); ax2.set_ylim(-0.6, 2.8)
    logticks(ax2, [0.75, 1, 1.5, 2]); ax2.set_xlim(0.7, 2.2)
    ax2.set_xlabel("Rate ratio, OA vs closed"); ax2.spines["left"].set_visible(True)
    panel(ax2, "b", dx=-0.02, dy=1.02)
    save(fig, "Fig7_H2a_visibility")


# ─────────────────────────── Fig. 8: periods ───────────────────────────
def fig8():
    per = ["2003\u201309", "2010\u201314", "2015\u201319", "2020\u201324"]; n = [483, 1616, 4541, 5380]
    un = [(2.109, 1.192, 3.732), (1.601, 1.091, 2.352), (1.285, 1.071, 1.543), (1.720, 1.284, 2.304)]
    ad = [(1.643, 1.026, 2.631), (1.274, 0.890, 1.824), (1.223, 1.014, 1.474), (1.605, 1.246, 2.068)]
    fig, ax = plt.subplots(figsize=(4.6, 3.5)); fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.34)
    for series, off, filled, lab in [(un, -0.08, True, "Unadjusted"), (ad, 0.08, False, "With academic-citation adjustment")]:
        xs = np.arange(4) + off
        for x, (rr, lo, hi) in zip(xs, series):
            ax.plot([x, x], [lo, hi], color=K, lw=1.1)
            ax.plot([x - .04, x + .04], [lo, lo], color=K, lw=1.1); ax.plot([x - .04, x + .04], [hi, hi], color=K, lw=1.1)
        ax.plot(xs, [s[0] for s in series], "o", ms=5.5, mfc=K if filled else "white", mec=K, mew=1.1, zorder=3, label=lab)
    ax.axhline(1, color=K, lw=0.7, ls=(0, (1, 2)))
    logticks(ax, [1, 1.5, 2, 3, 4], "y"); ax.set_ylim(0.85, 4.2); ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{p}\nN = {k:,}" for p, k in zip(per, n)])
    ax.set_ylabel("Rate ratio, OA vs closed (log scale)"); ax.set_xlabel("Publication period")
    ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    fig.text(0.16, 0.035, "Endpoint contrast (2015\u201319 \u00f7 2003\u201309, unadjusted), defined in the analysis plan\nfixed after peer review but before estimation: ratio 0.61 (95% CI 0.33\u20131.12; wild-bootstrap p = .16).\nLater periods have had less time to accumulate patent citations.",
             fontsize=7.2, va="bottom", ha="left")
    save(fig, "Fig8_H2b_periods")


# ─────────────────────────── Fig. 11: colours ───────────────────────────
def fig11():
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_axes([0.11, 0.20, 0.27, 0.72]); bx = fig.add_axes([0.58, 0.20, 0.18, 0.72])
    col = [("Gold", (1.330, 1.122, 1.575), (1.303, 1.104, 1.537)), ("Green", (2.110, 1.786, 2.493), (1.526, 1.291, 1.805)),
           ("Hybrid", (1.758, 1.311, 2.357), (1.187, 0.971, 1.451)), ("Bronze", (0.974, 0.770, 1.232), (0.876, 0.703, 1.091))]
    for i, (lab, m0, m1) in enumerate(col):
        y = 3 - i
        for (rr, lo, hi), off, filled in [(m0, 0.14, True), (m1, -0.14, False)]:
            yy = y + off
            ax.plot([lo, hi], [yy, yy], color=K, lw=1.1)
            ax.plot([lo, lo], [yy - .07, yy + .07], color=K, lw=1.1); ax.plot([hi, hi], [yy - .07, yy + .07], color=K, lw=1.1)
            ax.plot(rr, yy, "o", ms=5.5, mfc=K if filled else "white", mec=K, mew=1.1, zorder=3)
            ax.text(1.03, yy, fmt(rr, lo, hi), transform=ax.get_yaxis_transform(), fontsize=6.8, va="center")
    ax.axvline(1, color=K, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([c[0] for c in col]); ax.set_ylim(-0.6, 3.6)
    logticks(ax, [0.7, 1, 1.5, 2, 2.5]); ax.set_xlim(0.65, 2.7)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Rate ratio vs closed (log scale)")
    ax.plot([], [], "o", ms=5.5, mfc=K, mec=K, label="M0: colours only")
    ax.plot([], [], "o", ms=5.5, mfc="white", mec=K, mew=1.1, label="M1: + academic citations")
    ax.plot([], [], "s", ms=5.5, mfc="white", mec=K, mew=1.1, label="M2: + publication type, field tags (panel b)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=1, handletextpad=0.3)
    ax.text(1.03, 1.0, "RR (95% CI)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    panel(ax, "a", dx=-0.18, dy=1.02)
    con = [("Green / Gold", "M0", (1.587, 1.219, 2.067), 1), ("", "M1", (1.172, 0.894, 1.536), 0), ("", "M2", (1.096, 0.807, 1.489), 0),
           ("Bronze / Gold", "M0", (0.732, 0.589, 0.910), 1), ("", "M1", (0.672, 0.544, 0.831), 0)]
    ys = [5.0, 4.0, 3.0, 1.6, 0.6]
    tr = bx.get_yaxis_transform()
    for y, (grp, mod, (rr, lo, hi), f) in zip(ys, con):
        bx.plot([lo, hi], [y, y], color=K, lw=1.1)
        bx.plot([lo, lo], [y - .16, y + .16], color=K, lw=1.1); bx.plot([hi, hi], [y - .16, y + .16], color=K, lw=1.1)
        bx.plot(rr, y, "o" if mod != "M2" else "s", ms=5.5, mfc=K if f else "white", mec=K, mew=1.1, zorder=3)
        bx.text(-0.03, y, mod, transform=tr, ha="right", va="center", fontsize=8)
        bx.text(1.05, y, fmt(rr, lo, hi), transform=tr, va="center", fontsize=8)
        if grp:
            bx.text(-0.30, y + 0.62, grp, transform=tr, ha="left", va="center", fontsize=8, fontweight="bold")
    bx.axvline(1, color=K, lw=0.7, ls=(0, (3, 2)))
    logticks(bx, [0.5, 0.75, 1, 1.5, 2]); bx.set_xlim(0.5, 2.3); bx.set_ylim(-0.4, 6.0)
    bx.set_yticks([]); bx.spines["left"].set_visible(False)
    bx.set_xlabel("Ratio of rate ratios (log scale)")
    bx.text(1.05, 6.05, "Ratio (95% CI)", transform=tr, fontsize=8, fontweight="bold", va="bottom")
    bx.text(0.0, -0.36, "Ratio of Green/Gold rate ratios, M0 \u00f7 M1 = 1.36\n(cluster-bootstrap 95% CI 1.24\u20131.49; 300 draws).",
            transform=bx.transAxes, fontsize=7.2, va="top", ha="left")
    panel(bx, "b", dx=-0.42, dy=1.02)
    save(fig, "Fig11_H3_colour")


# ─────────────────────────── Fig. N2: Lens vs Unpaywall ───────────────────────────
def figN2():
    rows = ["Gold", "Green", "Hybrid", "Bronze", "Closed"]; cols = ["Gold", "Green", "Hybrid", "Bronze", "Closed"]
    M = np.array([[4569, 54, 196, 70, 139],       # Lens gold  -> Unpaywall gold,green,hybrid,bronze,closed
                  [11, 569, 36, 199, 255],        # Lens green
                  [10, 1, 359, 10, 4],            # Lens hybrid
                  [16, 3, 22, 147, 107],          # Lens bronze
                  [5, 104, 27, 93, 4844]])        # Lens closed
    tot = M.sum(1) + 0
    pct = M / tot[:, None] * 100
    fig, ax = plt.subplots(figsize=(4.9, 3.9)); fig.subplots_adjust(left=0.20, right=0.86, top=0.86, bottom=0.14)
    ax.imshow(pct, cmap="Greys", vmin=0, vmax=115, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = pct[i, j]; ax.text(j, i, f"{M[i, j]:,}\n({v:.0f}%)", ha="center", va="center", fontsize=7.5, color="white" if v >= 80 else K)
    for i in range(5):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=K, lw=1.8))
        ax.text(5.0, i, f"{tot[i]:,}", ha="left", va="center", fontsize=7.5, transform=ax.transData)
    ax.text(5.0, -0.85, "Row total", ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(cols); ax.set_yticks(range(5)); ax.set_yticklabels(rows)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Unpaywall classification"); ax.set_ylabel("Lens.org classification")
    for s in ax.spines.values(): s.set_visible(True)
    ax.tick_params(length=0)
    fig.text(0.20, 0.03, "Cells: number of records (paper\u2013institution pairs; row %). Diagonal outlined: exact agreement 88.1% (10,488 / 11,905 matched records,\nincl. 55 with Lens \u2018other\u2019 colour not shown; 88.5% of the 11,850 shown).",
             fontsize=7, va="bottom", ha="left")
    save(fig, "FigN2_lens_unpaywall")


# ─────────────────────────── Fig. N1: leave-one-institution-out ───────────────────────────
def figN1(path):
    import pandas as pd
    t = pd.read_csv(path)
    def clean(n):
        n = n.title().replace(" Of ", " of ").replace(" And ", " and ")
        return (n.replace("Xi An", "Xi\u2019an").replace("Sun Yat Sen", "Sun Yat-sen")
                 .replace("Soochow University Suzhou", "Soochow University (Suzhou)")
                 .replace("Shanghai Jiao Tong", "Shanghai Jiao Tong"))
    t["name"] = t["dropped"].map(clean)
    order = (t.drop_duplicates("dropped").sort_values("dropped_n", ascending=True)[["dropped", "name", "dropped_n"]]
             .reset_index(drop=True))                      # common order: institution size (records dropped)
    specs = [("H1 OA + lc", "H1: OA vs closed,\nadjusted"), ("H2a OA x lc_c", "H2a: OA \u00d7 visibility"),
             ("H3 Green/Gold M1", "H3: Green / Gold,\nadjusted (M1)")]
    full = {"H1 OA + lc": 1.338, "H2a OA x lc_c": 1.146, "H3 Green/Gold M1": 1.172}
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 7.6), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.90, bottom=0.125, wspace=0.08)
    for k, (ax, (sp, ttl)) in enumerate(zip(axes, specs)):
        d = t[t.spec == sp].set_index("dropped").loc[order.dropped].reset_index()
        ax.axvline(1, color=K, lw=0.5); ax.axvline(full[sp], color=K, lw=0.8, ls=(0, (3, 2)))
        for yi, r in zip(y, d.itertuples()):
            ax.plot([r.lo, r.hi], [yi, yi], color="0.6", lw=0.8, solid_capstyle="butt")
            ax.plot(r.RR, yi, "o", ms=3.8, mfc=K if r.p < .05 else "white", mec=K, mew=0.8, zorder=3)
        ax.set_ylim(-1.2, len(d) - 0.2)
        ax.set_title(ttl, fontsize=8.5, fontweight="bold", pad=14)
        ax.text(0.5, 1.01, f"{int((d.p < .05).sum())}/{len(d)} with p < .05  |  RR {d.RR.min():.2f}\u2013{d.RR.max():.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
        if k == 0:
            ax.set_yticks(y); ax.set_yticklabels([f"{n} ({m:,})" for n, m in zip(order.name, order.dropped_n)], fontsize=6.6)
        else:
            ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_xlabel(["Rate ratio without institution", "Interaction RR per log-unit\nwithout institution", "Rate ratio without institution"][k]); ax.tick_params(axis="x", labelsize=7)
        ax.text(-0.02 if k else -0.02, 1.075, "abc"[k], transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom") if k else ax.text(-0.02, 1.075, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
    fig.text(0.30, 0.004, "Institutions ordered by number of records dropped (in parentheses). Filled: p < .05; open: p \u2265 .05.\n"
             "Dashed line: full-sample estimate; solid line: RR = 1. Bars: 95% CI.", fontsize=6.8, va="bottom")
    save(fig, "FigN1_leave_one_institution_out")


if __name__ == "__main__":
    fig2(); fig6(); fig7(); fig8(); fig11(); figN2()
    if os.path.exists(C3):
        figN1(C3)
    else:
        print("Fig N1 skipped: put C3_leave_one_institution_out.csv next to the script or pass its path")