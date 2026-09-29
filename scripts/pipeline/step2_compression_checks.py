#!/usr/bin/env python3
"""
STEP 2 (exploratory) - Is the OA-premium compression real, or a citation-exposure / functional-form artefact?
=============================================================================================
Why: the sample is restricted to papers with >=1 citing patent, and recent papers have had less
time to accumulate patent citations. Compression of the log-scale OA gap could arise mechanically
(most recent papers sit at exactly 1). This script re-estimates the period-specific OA premium
under outcomes and samples that are robust to that concern.

Blocks
  S2-A  Descriptives by period x OA: mean/median patents, share ==1, >=2, >=3
  S2-B  Period-specific OA premium for 4 outcomes x 4 samples x (with/without academic citations)
          outcomes : log(1+patents) | P(>=2) LPM | P(>=3) LPM | PPML (rate ratio scale)
          samples  : full | open-at-publication (Gold/Hybrid vs closed) | pub<=2019 (>=5y exposure)
                     | pub<=2019 & Gold/Hybrid
  S2-C  Breakpoint OA x Post (2013, 2015) with wild-cluster bootstrap for LPM/log outcomes
  S2-D  Annual OA premium 2003-2024 (full and Gold/Hybrid) -> figure
  S2-E  H3 add-ons: Green-Gold contrast on P(>=2) and top-1% logit (65 'other' rows dropped,
          which caused separation in the previous run)

Run (same directory as step1_redesign_analysis.py, which is imported as the estimation engine):
  RV_RAW=./outputs/Transport_CN_Scholarly_Works_with_cluster.csv \
  RV_OUT=./outputs/step2 RV_BOOT=999 python3 -u step2_compression_checks.py
Output: RV_OUT/STEP2_REPORT.md, S2_*.csv, F_S2_annual_premium.png
"""
import os, sys
from pathlib import Path
os.environ.setdefault("RV_OUT", "./outputs/step2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import step1_redesign_analysis as base   # renamed from redesign_analysis_and_blueprint.py

OUT = base.OUT
YMIN, YMAX = base.YMIN, base.YMAX
PERIODS = base.PERIODS            # [(2003,2009),(2010,2014),(2015,2019),(2020,2024)]
ROWS = []


# ───────────── PPML with the same result structure as base.fit ─────────────
def ppml_fit(d, y, xvars, fe=("cluster", "year")):
    d = d.dropna(subset=xvars).copy()
    M = base.design(d, xvars, fe); names = list(M.columns)
    gi = pd.factorize(d["cluster"])[0]; G = gi.max() + 1
    m = sm.GLM(d[y].astype(float).values, M.values, family=sm.families.Poisson()) \
        .fit(cov_type="cluster", cov_kwds={"groups": gi})
    b = np.asarray(m.params); V = np.asarray(m.cov_params()); crit = ss.t.ppf(.975, G - 1)
    coefs = {}
    for t in xvars:
        if t in names:
            j = names.index(t); se = np.sqrt(V[j, j])
            coefs[t] = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se,
                            p=2 * ss.t.sf(abs(b[j] / se), G - 1), p_wcb=np.nan)
    return dict(b=b, V=V, names=names, G=G, n=len(d), coefs=coefs)


def logit_fit(d, y, xvars, fe=("year",)):
    d = d.dropna(subset=xvars).copy()
    M = base.design(d, xvars, fe); names = list(M.columns)
    gi = pd.factorize(d["cluster"])[0]; G = gi.max() + 1
    m = sm.Logit(d[y].values.astype(float), M.values).fit(disp=0, cov_type="cluster", cov_kwds={"groups": gi})
    b = np.asarray(m.params); V = np.asarray(m.cov_params()); crit = ss.t.ppf(.975, G - 1)
    coefs = {}
    for t in xvars:
        if t in names:
            j = names.index(t); se = np.sqrt(V[j, j])
            coefs[t] = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se,
                            p=2 * ss.t.sf(abs(b[j] / se), G - 1), p_wcb=np.nan)
    return dict(b=b, V=V, names=names, G=G, n=len(d), coefs=coefs)


def add(block, sample, outcome, lc, term, r, n):
    ROWS.append(dict(block=block, sample=sample, outcome=outcome, lc=lc, term=term, n=n,
                     **{k: r[k] for k in ("beta", "se", "lo", "hi", "p", "p_wcb")}))


# ───────────── data ─────────────
def prepare():
    Dall = base.load()
    D = Dall[Dall.year.between(YMIN, YMAX)].copy()
    D["ge2"] = (D["Citing_Patents"] >= 2).astype(float)
    D["ge3"] = (D["Citing_Patents"] >= 3).astype(float)
    D["period"] = pd.cut(D["year"], [a - 1 for a, _ in PERIODS] + [PERIODS[-1][1]],
                         labels=[f"{a}-{b}" for a, b in PERIODS])
    Dgh = D[D.colour.isin(["closed", "gold", "hybrid"])].copy()
    Dgh["oa_pub"] = Dgh.colour.isin(["gold", "hybrid"]).astype(int)
    return D, Dgh


# ───────────── S2-A ─────────────
def s2a(D):
    g = D.groupby(["period", "oa"]).agg(
        n=("lp", "size"), mean_patents=("Citing_Patents", "mean"), median_patents=("Citing_Patents", "median"),
        share_eq1=("Citing_Patents", lambda s: (s == 1).mean()), share_ge2=("ge2", "mean"),
        share_ge3=("ge3", "mean"), mean_lc=("lc", "mean")).reset_index()
    g.to_csv(OUT / "S2A_descriptives_period_by_oa.csv", index=False)
    return g


# ───────────── S2-B ─────────────
def s2b(D, Dgh):
    samples = {
        "full": (D, "oa"),
        "open-at-publication (Gold/Hybrid vs closed)": (Dgh, "oa_pub"),
        "pub<=2019 (>=5y exposure)": (D[D.year <= 2019], "oa"),
        "pub<=2019 & Gold/Hybrid": (Dgh[Dgh.year <= 2019], "oa_pub"),
    }
    outcomes = {"log(1+patents)": ("lp", base.fit), "P(>=2) LPM": ("ge2", base.fit),
                "P(>=3) LPM": ("ge3", base.fit), "PPML patents": ("Citing_Patents", ppml_fit)}
    summ = []
    for sname, (d0, oacol) in samples.items():
        d = d0.copy()
        for i, (a, b) in enumerate(PERIODS):
            d[f"oaP{i}"] = d[oacol] * d["year"].between(a, b).astype(int)
        for oname, (y, fn) in outcomes.items():
            for lc in (False, True):
                xv = [f"oaP{i}" for i in range(len(PERIODS))] + (["lc"] if lc else [])
                try:
                    res = fn(d, y, xv)
                except Exception as e:
                    print(f"  skip {sname}|{oname}|lc={lc}: {e}"); continue
                present = [i for i in range(len(PERIODS)) if f"oaP{i}" in res["coefs"]]
                for i in present:
                    add("S2-B", sname, oname, lc, f"oaP{i}", res["coefs"][f"oaP{i}"], res["n"])
                pairs = []
                if len(present) >= 2:
                    pairs.append(("last_minus_first", present[-1], present[0]))
                if 2 in present and 0 in present and present[-1] != 2:
                    pairs.append(("P2_minus_P0 (2015-19 vs 2003-09)", 2, 0))
                for nm, hi_, lo_ in pairs:
                    r = base.lincom(res, {f"oaP{hi_}": 1, f"oaP{lo_}": -1})
                    add("S2-B", sname, oname, lc, nm, r, res["n"])
                    summ.append(dict(sample=sname, outcome=oname, lc=lc, contrast=nm, beta=r["beta"],
                                     lo=r["lo"], hi=r["hi"], p=r["p"]))
    s = pd.DataFrame(summ); s.to_csv(OUT / "S2B_compression_summary.csv", index=False)
    return s


# ───────────── S2-C ─────────────
def s2c(D, Dgh):
    out = []
    for sname, d0, oacol in [("full", D, "oa"), ("Gold/Hybrid", Dgh, "oa_pub")]:
        for cut in (2013, 2015):
            d = d0.copy(); d["oa_post"] = d[oacol] * (d["year"] >= cut).astype(int)
            for oname, y, fn in [("log(1+patents)", "lp", base.fit), ("P(>=2) LPM", "ge2", base.fit),
                                 ("PPML patents", "Citing_Patents", ppml_fit)]:
                for lc in (False, True):
                    xv = [oacol, "oa_post"] + (["lc"] if lc else [])
                    try:
                        res = fn(d, y, xv, boot=("oa_post",)) if fn is base.fit else fn(d, y, xv)
                    except TypeError:
                        res = fn(d, y, xv)
                    except Exception as e:
                        print("  skip", sname, cut, oname, e); continue
                    r = res["coefs"].get("oa_post")
                    if r:
                        add("S2-C", sname, oname, lc, f"OAxPost{cut}", r, res["n"])
                        out.append(dict(sample=sname, cut=cut, outcome=oname, lc=lc, **r))
    t = pd.DataFrame(out); t.to_csv(OUT / "S2C_breakpoint.csv", index=False); return t


# ───────────── S2-D ─────────────
def s2d(D, Dgh):
    fig, ax = plt.subplots(figsize=(9, 4.8)); rows = []
    for (sname, d0, oacol), col in zip([("full", D, "oa"), ("Gold/Hybrid vs closed", Dgh, "oa_pub")],
                                       ["#1B5EA8", "#C0392B"]):
        d = d0.copy(); yrs = sorted(d.year.unique()); xv = []
        for y_ in yrs:
            d[f"oa_y{y_}"] = d[oacol] * (d.year == y_).astype(int); xv.append(f"oa_y{y_}")
        res = base.fit(d, "lp", xv)
        yy, bb, lo, hi = [], [], [], []
        for y_ in yrs:
            r = res["coefs"].get(f"oa_y{y_}")
            if r:
                yy.append(y_); bb.append(r["beta"]); lo.append(r["lo"]); hi.append(r["hi"])
                rows.append(dict(sample=sname, year=y_, **r))
        ax.errorbar(np.array(yy) + (0.1 if "Gold" in sname else -0.1), bb,
                    yerr=[np.array(bb) - np.array(lo), np.array(hi) - np.array(bb)],
                    fmt="o-", ms=4, capsize=2, color=col, label=sname)
    ax.axhline(0, color="grey", ls=":"); ax.set_ylabel("OA − closed, log(1+patents), no lc")
    ax.set_title("Annual OA premium (95% CI); a steady slide toward the last years suggests exposure truncation")
    ax.title.set_fontsize(9); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "F_S2_annual_premium.png", dpi=200); plt.close()
    t = pd.DataFrame(rows); t.to_csv(OUT / "S2D_annual_premium.csv", index=False); return t


# ───────────── S2-E ─────────────
def s2e(D):
    d = D[D.colour != "other"].copy()
    cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"]; out = []
    for oname, y, fn, fe in [("P(>=2) LPM", "ge2", base.fit, ("cluster", "year")),
                             ("top-1% logit", "top1", logit_fit, ("year",)),
                             ("log(1+patents)", "lp", base.fit, ("cluster", "year"))]:
        if y == "top1":
            d["top1"] = (d["Citing_Patents"] >= d["Citing_Patents"].quantile(.99)).astype(float)
        for lc in (False, True):
            xv = cols + (["lc"] if lc else [])
            try:
                res = fn(d, y, xv, fe)
            except Exception as e:
                print("  skip S2-E", oname, e); continue
            gg = base.lincom(res, {"c_green": 1, "c_gold": -1})
            add("S2-E", "colour (other dropped)", oname, lc, "green_minus_gold", gg, res["n"])
            out.append(dict(outcome=oname, lc=lc, **gg))
    t = pd.DataFrame(out); t.to_csv(OUT / "S2E_green_gold_addons.csv", index=False); return t


# ───────────── report ─────────────
def f3(x): return f"{x:.3f}"
def fp(p): return "<.001" if p < .001 else f"{p:.3f}"


def report(desc, summ, brk, ann, add_):
    L = []; w = L.append
    w("# STEP 2 report — is the OA-premium compression an exposure/functional-form artefact?\n")
    w(f"Sample {YMIN}-{YMAX}. Clusters=25; CIs use t(G-1). Wild-cluster bootstrap p only for OA x Post in LPM/log models.\n")
    w("## S2-A Descriptives (period x OA)\n"); w(desc.round(3).to_markdown(index=False)); w("")
    w("## S2-B Compression summary (premium in later period minus earlier)\n")
    t = summ.copy()
    for c in ("beta", "lo", "hi"): t[c] = t[c].map(f3)
    t["p"] = t["p"].map(fp)
    w(t.to_markdown(index=False)); w("")
    w("### Auto-reading guide\n")
    key = summ[(summ.contrast.str.startswith("P2_minus_P0")) | ((summ.contrast == "last_minus_first") & summ["sample"].str.contains("2019"))]
    n_neg = int(((key.beta < 0) & (key.p < .05)).sum()); n_all = len(key)
    w(f"- Among {n_all} exposure-comparable contrasts (2015-19 vs 2003-09; all papers >=5y old), {n_neg} are negative and p<.05.")
    lp = key[(key.outcome == "log(1+patents)") & (key["sample"] == "full")]
    pp = key[(key.outcome == "PPML patents") & (key["sample"] == "full")]
    if len(lp): w("- log outcome, full sample, 2015-19 minus 2003-09: " + "; ".join(
        f"lc={r.lc}: {f3(r.beta)} (p={fp(r.p)})" for r in lp.itertuples()))
    if len(pp): w("- PPML (ratio scale), same contrast: " + "; ".join(
        f"lc={r.lc}: {f3(r.beta)} (p={fp(r.p)})" for r in pp.itertuples()))
    w("- If the negative contrast survives in the pub<=2019 samples and in Gold/Hybrid-only, truncation and retrospective Green coding "
      "do not explain it. If it vanishes there, drop the compression claim.\n")
    w("## S2-C Breakpoint OA x Post\n")
    b = brk.copy()
    for c in ("beta", "se", "lo", "hi"): b[c] = b[c].map(f3)
    b["p"] = b["p"].map(fp); b["p_wcb"] = b["p_wcb"].map(lambda x: "" if pd.isna(x) else fp(x))
    w(b[["sample", "cut", "outcome", "lc", "beta", "se", "p", "p_wcb"]].to_markdown(index=False)); w("")
    w("## S2-D Annual premium\nSee F_S2_annual_premium.png and S2D_annual_premium.csv.\n")
    w("## S2-E Green minus Gold (other dropped)\n")
    a = add_.copy()
    for c in ("beta", "se", "lo", "hi"): a[c] = a[c].map(f3)
    a["p"] = a["p"].map(fp)
    w(a[["outcome", "lc", "beta", "lo", "hi", "p"]].to_markdown(index=False)); w("")
    w("Note: these contrasts use CRV1 with t(G-1); no bootstrap. Treat p between .01 and .10 as suggestive.\n")
    (OUT / "STEP2_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    D, Dgh = prepare()
    print(f"main N={len(D):,}; Gold/Hybrid-vs-closed N={len(Dgh):,}")
    print("S2-A ..."); desc = s2a(D)
    print("S2-B ..."); summ = s2b(D, Dgh)
    print("S2-C ..."); brk = s2c(D, Dgh)
    print("S2-D ..."); ann = s2d(D, Dgh)
    print("S2-E ..."); add_ = s2e(D)
    pd.DataFrame(ROWS).to_csv(OUT / "S2_ALL_COEFFICIENTS.csv", index=False)
    report(desc, summ, brk, ann, add_)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
