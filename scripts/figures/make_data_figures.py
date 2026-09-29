#!/usr/bin/env python3
"""
Data-driven, journal-style black-and-white figures (run locally on the raw data).

  Fig. 3   UMAP of the 25 research clusters        (needs embeddings; cached after first run)
  Fig. 4   annual patent-cited papers + OA share    (2003-2024)
  Fig. 11A OA colour composition by period          (Lens 'Open Access Colour')
  Fig. 14A academic citations at extraction by colour
  Fig. A1  citing patents per record (OA vs closed; mean by colour)

Usage (from ~/Research/Scientometrics_R1):
  python3 make_data_figures.py \
      --raw ./Transport_CN_Scholarly_Works.csv \
      --clustered ./outputs/Transport_CN_Scholarly_Works_with_cluster.csv \
      --out ./figures_data
  add  --skip-umap   to make only Fig. 4 / 11A / 14A / A1 (no sentence-transformers needed)

Sample rule = audit trail of the revised paper:
  drop blank rows -> drop missing patent count -> drop missing year and 2026 -> 2003-2024
  (expected: 12,302 -> 12,020 records; the script warns if the counts differ).
Every number in the figures is computed from the data; nothing is hard-coded.
"""
import os, re, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="./Transport_CN_Scholarly_Works.csv")
ap.add_argument("--clustered", default="./outputs/Transport_CN_Scholarly_Works_with_cluster.csv")
ap.add_argument("--out", default="./figures_data")
ap.add_argument("--skip-umap", action="store_true")
ap.add_argument("--umap-cache", default=None, help="npy with 2-D coordinates (rows = filtered raw order)")
ap.add_argument("--encoding", default="MacRoman")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.unicode_minus": True,
})
K = "black"
PERIODS = [(2003, 2009, "2003\u201309"), (2010, 2014, "2010\u201314"), (2015, 2019, "2015\u201319"), (2020, 2024, "2020\u201324")]
COLOURS = ["Gold", "Green", "Hybrid", "Bronze", "Closed"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig); print("saved", name)


# ───────────────────────── data ─────────────────────────
def load(path, enc=None):
    raw = pd.read_csv(path, encoding=enc or args.encoding, low_memory=False)
    raw.columns = raw.columns.str.strip()
    n0 = len(raw)
    d = raw.dropna(how="all").copy()
    d["patents"] = pd.to_numeric(d["Citing Patents Count"], errors="coerce")
    d = d.dropna(subset=["patents"]).copy()
    d["year"] = pd.to_numeric(d["Publication Year"], errors="coerce")
    d = d[d["year"].notna() & (d["year"] != 2026)].copy()
    n_all = len(d)
    d = d.reset_index(drop=True)
    d["year"] = d["year"].astype(int)
    d["oa"] = d["Is Open Access"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(int)
    d["acad"] = pd.to_numeric(d.get("Citing Works Count"), errors="coerce")
    col = d.get("Open Access Colour", pd.Series(np.nan, index=d.index)).astype(str).str.strip().str.lower()
    m = {"gold": "Gold", "green": "Green", "hybrid": "Hybrid", "bronze": "Bronze"}
    d["colour"] = col.map(m)
    d.loc[d["colour"].isna() & d["oa"].eq(0), "colour"] = "Closed"
    d["colour"] = d["colour"].fillna("Other")          # OA but unclassified colour -> excluded from colour figures
    print(f"raw rows {n0:,} -> {n_all:,} records (expected 12,302)")
    main = d[(d["year"] >= 2003) & (d["year"] <= 2024)].copy()
    print(f"main window 2003-2024: {len(main):,} records (expected 12,020)")
    if n_all != 12302 or len(main) != 12020:
        print("  WARNING: counts differ from the revised audit trail; check the input file version.")
    return d, main


def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ───────────────────────── Fig. 4 ─────────────────────────
def fig4(main):
    g = main.groupby("year").agg(n=("oa", "size"), k=("oa", "sum")).reindex(range(2003, 2025), fill_value=0)
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.12))
    a.bar(g.index, g.n, width=0.72, color="0.72", edgecolor=K, lw=0.6)
    for x, v in zip(g.index, g.n):
        dx = 0.30 if x == 2015 else 0.0        # label of the 2015 bar sits beside the cut-off line, not on it
        a.text(x + dx, v + g.n.max() * 0.015, f"{v:,}", ha="center", va="bottom", fontsize=6.3, rotation=90, zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))
    a.set_ylabel("Patent-cited papers\n(paper\u2013institution records)"); a.set_ylim(0, g.n.max() * 1.22)
    for ax in (a, b): ax.axvline(2015, color=K, lw=0.8, ls=(0, (3, 2)))
    a.text(2015.15, g.n.max() * 1.17, "2015 period cut-off", fontsize=7, va="top", ha="left")
    share = g.k / g.n.replace(0, np.nan)
    lo, hi = zip(*[wilson(k, n) for k, n in zip(g.k, g.n)])
    small = g.n < 20
    b.vlines(g.index, lo, hi, color=K, lw=0.9)
    b.plot(g.index, share, "-", color=K, lw=0.7)
    b.plot(g.index[~small], share[~small], "o", ms=4.5, mfc=K, mec=K)
    b.plot(g.index[small], share[small], "o", ms=4.5, mfc="white", mec=K, mew=0.9)
    b.set_ylim(0, 1); b.set_ylabel("OA share (95% Wilson CI)")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_xlabel("Publication year"); b.set_xticks(range(2003, 2025, 2)); b.set_xlim(2002.4, 2024.6)
    b.plot([], [], "o", ms=4.5, mfc=K, mec=K, label="n \u2265 20"); b.plot([], [], "o", ms=4.5, mfc="white", mec=K, label="n < 20 (small cell)")
    b.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.3)
    a.text(-0.10, 1.02, "a", transform=a.transAxes, fontsize=11, fontweight="bold")
    b.text(-0.10, 1.02, "b", transform=b.transAxes, fontsize=11, fontweight="bold")
    g.assign(oa_share=share, ci_lo=lo, ci_hi=hi).to_csv(os.path.join(args.out, "Fig4_data.csv"))
    save(fig, "Fig4_annual_papers")


# ───────────────────────── Fig. 11A ─────────────────────────
def fig11a(main):
    d = main[main["colour"].isin(COLOURS)].copy()
    n_excl = len(main) - len(d)
    rows, ns = [], []
    for lo, hi, lab in PERIODS + [(2003, 2024, "All")]:
        s = d[(d["year"] >= lo) & (d["year"] <= hi)]
        rows.append(s["colour"].value_counts(normalize=True).reindex(COLOURS, fill_value=0) * 100); ns.append(len(s))
    P = pd.DataFrame(rows, index=[p[2] for p in PERIODS] + ["All"])
    P["n"] = ns; P.to_csv(os.path.join(args.out, "Fig11A_data.csv"))
    style = {"Gold": dict(fc="0.15", hatch=None, tc="white"), "Green": dict(fc="0.50", hatch=None, tc="white"),
             "Hybrid": dict(fc="white", hatch="////", tc=K), "Bronze": dict(fc="0.80", hatch="....", tc=K),
             "Closed": dict(fc="0.94", hatch=None, tc=K)}
    fig, ax = plt.subplots(figsize=(5.6, 3.6)); fig.subplots_adjust(left=0.10, right=0.76, bottom=0.28, top=0.95)
    x = np.arange(len(P)); x = np.where(x == 4, 4.35, x)
    bottom = np.zeros(len(P))
    for c in COLOURS:
        v = P[c].values; st = style[c]
        ax.bar(x, v, bottom=bottom, width=0.68, fc=st["fc"], hatch=st["hatch"], ec=K, lw=0.7, label=c if c != "Closed" else "Closed / unknown")
        for xi, vi, bi in zip(x, v, bottom):
            if vi >= 4:
                hb = st["hatch"] is not None      # hatched/dotted segments: black text on a white plate
                ax.text(xi, bi + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=7, color=K if hb else st["tc"], zorder=6,
                        bbox=dict(fc="white", ec="none", pad=0.8) if hb else None)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([f"{i}\nN = {n:,}" for i, n in zip(P.index, ns)])
    ax.set_ylim(0, 100); ax.set_ylabel("Share of records (%)"); ax.set_xlabel("Publication period")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=1.6)
    fig.text(0.10, 0.01, f"Records with Lens colour Gold/Green/Hybrid/Bronze or closed; {n_excl:,} records with other/unclassified colour omitted. "
             "Periods are descriptive.", fontsize=6.6, va="bottom", ha="left", wrap=True)
    save(fig, "Fig11A_colour_composition")


# ───────────────────────── Fig. 14A ─────────────────────────
def fig14a(main):
    d = main[main["colour"].isin(COLOURS) & main["acad"].notna()].copy()
    d = d.drop_duplicates("Lens ID")   # one row per paper
    n_omit = main["Lens ID"].nunique() - len(d)
    HIGHLIGHT_GREEN = False            # set True to shade the Green box (then say why in the caption)
    d["lc"] = np.log1p(d["acad"])
    data = [d.loc[d["colour"] == c, "lc"].values for c in COLOURS]
    kw = stats.kruskal(*data)
    fig, ax = plt.subplots(figsize=(4.8, 3.6)); fig.subplots_adjust(left=0.14, right=0.98, bottom=0.30, top=0.95)
    bp = ax.boxplot(data, positions=range(5), widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color=K, lw=1.4), boxprops=dict(lw=0.8, ec=K), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for p, c in zip(bp["boxes"], COLOURS): p.set_facecolor("0.80" if (c == "Green" and HIGHLIGHT_GREEN) else "white")
    means = [x.mean() for x in data]
    ax.plot(range(5), means, "D", ms=5, mfc=K, mec=K, zorder=4, label="Mean")
    for i, (m, x) in enumerate(zip(means, data)):
        ax.text(i + 0.33, m, f"{m:.2f}", fontsize=7, va="center", ha="left")
    ax.set_xticks(range(5)); ax.set_xticklabels([("Closed/\nunknown" if c == "Closed" else c) + f"\nn = {len(x):,}" for c, x in zip(COLOURS, data)])
    ax.set_ylabel("Academic citations at extraction,\nlog(1 + citations)")
    ax.plot([], [], color=K, lw=1.4, label="Median (box: IQR; whiskers: 5th\u201395th pct.)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, handlelength=1.6, ncol=1)
    ax.text(0.01, 1.0, f"Kruskal\u2013Wallis H = {kw.statistic:.1f}, p {'< .001' if kw.pvalue < .001 else '= %.3f' % kw.pvalue}\n(one row per paper, n = {len(d):,};\n{n_omit:,} papers with other/unclassified colour or missing citations omitted)",
            transform=ax.transAxes, fontsize=7, ha="left", va="top")
    ax.set_ylim(top=8.6)
    pd.DataFrame({"colour": COLOURS, "n": [len(x) for x in data], "mean_lc": means,
                  "median_lc": [np.median(x) for x in data]}).to_csv(os.path.join(args.out, "Fig14A_data.csv"), index=False)
    save(fig, "Fig14A_academic_citations_by_colour")


# ───────────────────────── Fig. A1 ─────────────────────────
def figA1(main):
    rng = np.random.default_rng(2025)
    p_oa = main.loc[main["oa"] == 1, "patents"].values; p_cl = main.loc[main["oa"] == 0, "patents"].values
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 9, "5\u20139"), (10, 10 ** 9, "\u2265 10")]
    share = lambda v: [100 * ((v >= lo) & (v <= hi)).mean() for lo, hi, _ in bins]
    so, sc = share(p_oa), share(p_cl); one_all = 100 * (main["patents"] == 1).mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.34))
    x = np.arange(len(bins)); w = 0.36
    a.bar(x - w / 2, so, w, fc="0.25", ec=K, lw=0.7, label=f"OA (n = {len(p_oa):,})")
    a.bar(x + w / 2, sc, w, fc="white", ec=K, lw=0.7, hatch="////", label=f"Closed / unknown (n = {len(p_cl):,})")
    for xi, vo, vc in zip(x, so, sc):
        a.text(xi - w / 2, vo + 0.8, f"{vo:.0f}", ha="center", va="bottom", fontsize=6.8)
        a.text(xi + w / 2, vc + 0.8, f"{vc:.0f}", ha="center", va="bottom", fontsize=6.8)
    a.set_xticks(x); a.set_xticklabels([t[2] for t in bins]); a.set_ylim(0, 85)
    a.set_xlabel("Citing patents per record"); a.set_ylabel("Share of records within group (%)")
    a.legend(loc="upper right", frameon=False, handlelength=1.6)
    a.text(0.97, 0.60, f"Exactly one citing patent:\nOA {so[0]:.0f}%, closed {sc[0]:.0f}%\nAll records: {one_all:.1f}%",
           transform=a.transAxes, ha="right", va="top", fontsize=7.5)
    dd = main[main["colour"].isin(COLOURS)]; allm = main["patents"].mean(); rows = []
    b.axhline(allm, color=K, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for i, c in enumerate(COLOURS):
        v = dd.loc[dd["colour"] == c, "patents"].values; m = v.mean()
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)]); lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append((c, len(v), m, lo, hi))
        b.plot([i, i], [lo, hi], color=K, lw=1.1, zorder=2)
        for yy in (lo, hi): b.plot([i - .07, i + .07], [yy, yy], color=K, lw=1.1, zorder=2)
        b.plot(i, m, "D", ms=6, mfc="white" if c == "Closed" else K, mec=K, mew=1.1, zorder=4)
        near = abs(m - allm) < 0.15            # keep the value label off the dashed mean line
        b.text(i + 0.14, m - 0.12 if near else m, f"{m:.2f}", ha="left", va="top" if near else "center", fontsize=7.5, zorder=5,
               bbox=dict(fc="white", ec="none", pad=0.5))
    b.set_xticks(range(5)); b.set_xticklabels([("Closed/\nunknown" if r[0] == "Closed" else r[0]) + f"\nn = {r[1]:,}" for r in rows])
    b.set_xlim(-0.6, 4.7); b.set_ylim(1.3, 4.7); b.tick_params(axis="x", labelsize=6.8)
    b.set_ylabel("Mean citing patents per record\n(95% bootstrap CI)")
    b.plot([], [], color=K, lw=0.8, ls=(0, (3, 2)), label=f"All records (mean {allm:.2f})")
    b.plot([], [], "D", ms=6, mfc="white", mec=K, mew=1.1, ls="none", label="Closed / unknown (reference)")
    b.legend(loc="upper right", frameon=False, handlelength=1.8)
    a.text(-0.16, 1.02, "a", transform=a.transAxes, fontsize=11, fontweight="bold")
    b.text(-0.20, 1.02, "b", transform=b.transAxes, fontsize=11, fontweight="bold")
    pd.DataFrame(rows, columns=["colour", "n", "mean", "ci_lo", "ci_hi"]).assign(all_mean=allm, one_patent_pct_all=one_all
        ).to_csv(os.path.join(args.out, "FigA1_data.csv"), index=False)
    save(fig, "FigA1_patent_counts")


# ───────────────────────── Fig. 3 ─────────────────────────
def get_umap2d(dfc):
    cache = args.umap_cache or os.path.join(args.out, "umap2d_cache.npy")
    if os.path.exists(cache):
        xy = np.load(cache); print("UMAP cache loaded:", cache)
        if len(xy) == len(dfc): return xy
        print("  cache length mismatch -> recomputing")
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import umap
    SEED = 2025; np.random.seed(SEED)
    txt = (dfc.get("Title", "").fillna("") + " " + dfc.get("Abstract", "").fillna("") + " " +
           dfc.get("Fields of Study", "").fillna("") + " " + dfc.get("Keywords", "").fillna("")).str.strip().tolist()
    txt = [t if t else "transport china research" for t in txt]
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(txt, batch_size=64, show_progress_bar=True,
                                                                             normalize_embeddings=True, convert_to_numpy=True)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(emb).astype(np.float32)
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb)
    np.save(cache, xy); return xy


def fig3(dfc):
    xy = get_umap2d(dfc)
    keep = ((dfc["year"] >= 2003) & (dfc["year"] <= 2024)).values
    d, xy = dfc[keep].copy(), xy[keep]
    cl = d["cluster"].astype(int).values; ids = sorted(set(cl))
    shades = ["0.25", "0.50", "0.70", "0.85"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for i, c in enumerate(ids):
        m = cl == c
        ax.scatter(xy[m, 0], xy[m, 1], s=2.2, c=shades[i % 4], lw=0, rasterized=True)
    for c in ids:
        m = cl == c; cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax.text(cx, cy, f"C{c}", fontsize=7.5, fontweight="bold", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"): ax.spines[s].set_visible(True)
    ax.text(0.01, 0.01, f"N = {len(d):,} records; {len(ids)} clusters. Shading alternates only to separate neighbours.",
            transform=ax.transAxes, fontsize=6.8, va="bottom")
    save(fig, "Fig3_UMAP_clusters")
    # top terms per cluster (for Appendix Table A2)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [" ".join((d.loc[cl == c, "Title"].fillna("")).tolist()) for c in ids]
        v = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 1), min_df=2)
        X = v.fit_transform(docs); terms = np.array(v.get_feature_names_out())
        pd.DataFrame({"cluster": ids, "n": [int((cl == c).sum()) for c in ids],
                      "top_terms": [", ".join(terms[np.argsort(-X[i].toarray()[0])[:5]]) for i in range(len(ids))]}
                     ).to_csv(os.path.join(args.out, "TableA2_cluster_top_terms.csv"), index=False)
        print("saved TableA2_cluster_top_terms.csv")
    except Exception as e:
        print("top-term table skipped:", e)


if __name__ == "__main__":
    _, main = load(args.raw)
    fig4(main); fig11a(main); fig14a(main); figA1(main)
    if not args.skip_umap:
        if not os.path.exists(args.clustered):
            print("clustered file not found -> Fig. 3 skipped"); sys.exit(0)
        dfc, _ = load(args.clustered, enc="utf-8-sig")   # written by phase0 script as utf-8-sig
        if "cluster" not in dfc.columns: print("no 'cluster' column -> Fig. 3 skipped"); sys.exit(0)
        fig3(dfc)