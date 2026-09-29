#!/usr/bin/env python3
"""
REDESIGN ANALYSIS + MANUSCRIPT BLUEPRINT (exploratory round 1; log(1+y) outcome, OLS with CRV1 + wild-cluster bootstrap)
======================================================
Superseded as the primary analysis by scripts/pipeline/step3..5 (zero-truncated count models).
NOTE: the estimation code is unchanged from the run that produced results/exploratory/*; only the
write_report() prose (section map, reviewer map, "must be written outside this script" lists) was
condensed for this repository, so REDESIGN_BLUEPRINT.md regenerated here is shorter than the archived one.
Kept for transparency: it documents the first-round results that motivated redefining H2/H3.

Evidence grades used throughout
  DESCRIPTIVE | ADJUSTED ASSOCIATION | STATISTICAL DECOMPOSITION | QUASI-CAUSAL (not run here)

Input
  RV_RAW : Transport_CN_Scholarly_Works_with_cluster.csv  (raw columns + `cluster`,
           needs OA, Pub_Year, Citing_Patents, Citing_Works, cluster, Open Access Colour)
Env
  RV_OUT (default ./outputs/redesign), RV_YEAR_MIN (2003), RV_YEAR_MAX (2024),
  RV_BOOT (999 wild-cluster bootstrap draws), RV_BOOT_MED (300 mediation draws)

Inference note: only 25 clusters -> SEs are CRV1 with t(G-1) critical values, and the
key coefficients also get a restricted wild-cluster (Rademacher) bootstrap p-value.
Requires: pandas numpy scipy statsmodels matplotlib tabulate
"""
import os, re, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.sparse as sp
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RAW = Path(os.environ.get("RV_RAW", "./outputs/Transport_CN_Scholarly_Works_with_cluster.csv"))
OUT = Path(os.environ.get("RV_OUT", "./outputs/redesign")); OUT.mkdir(parents=True, exist_ok=True)
YMIN = int(os.environ.get("RV_YEAR_MIN", 2003)); YMAX = int(os.environ.get("RV_YEAR_MAX", 2024))
B_BOOT = int(os.environ.get("RV_BOOT", 999)); B_MED = int(os.environ.get("RV_BOOT_MED", 300))
SEED = 2025
RNG = np.random.default_rng(SEED)
PERIODS = [(2003, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]
BREAKS = [2012, 2013, 2014, 2015, 2016]
ROWS = []      # every coefficient reported anywhere
R = {}         # results store for the report


# ───────────────────────── data ─────────────────────────
def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "MacRoman", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("cannot read " + str(path))


def load():
    d = read_csv(RAW)
    d.columns = d.columns.str.strip()
    need = ["Pub_Year", "Citing_Patents", "Citing_Works", "cluster"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"missing columns {miss}; use the *_with_cluster.csv file")
    if "OA" not in d.columns:
        d["OA"] = d["Is Open Access"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["oa"] = d["OA"].astype(int)
    d["year"] = pd.to_numeric(d["Pub_Year"], errors="coerce")
    d = d.dropna(subset=["year", "Citing_Patents", "cluster"]).copy()
    d["year"] = d["year"].astype(int)
    d["lp"] = np.log1p(d["Citing_Patents"].astype(float))
    d["lc"] = np.log1p(pd.to_numeric(d["Citing_Works"], errors="coerce").fillna(0))
    col = d["Open Access Colour"].fillna("").astype(str).str.strip().str.lower() \
        if "Open Access Colour" in d.columns else pd.Series("", index=d.index)
    d["colour"] = np.where(d["oa"] == 0, "closed",
                           np.where(col.isin(["gold", "green", "hybrid", "bronze"]), col, "other"))
    for c in ["gold", "green", "hybrid", "bronze", "other"]:
        d["c_" + c] = (d["colour"] == c).astype(int)
    pt = d["Pub_Type"] if "Pub_Type" in d.columns else d.get("Publication Type", pd.Series("unk", index=d.index))
    d["pubtype"] = pt.fillna("unk").astype(str).str.lower()
    fos = d["Fields of Study"].fillna("") if "Fields of Study" in d.columns else pd.Series("", index=d.index)
    d["ln_fields"] = np.log1p(fos.apply(lambda s: len([x for x in re.split(r"[;,]", s) if x.strip()])))
    d["cluster"] = d["cluster"].astype(int).astype(str)
    return d.reset_index(drop=True)


def add_lag_share(d):
    """Cluster OA share among papers published in t-3..t-1 (needs >=10 papers).
    Not mechanically linked to the paper's own OA status. Still retrospective OA coding."""
    out = []
    yrs = range(int(d.year.min()), int(d.year.max()) + 1)
    for c, g in d.groupby("cluster"):
        t = g.groupby("year")["oa"].agg(s="sum", n="count").reindex(yrs, fill_value=0)
        s = t["s"].rolling(3, min_periods=3).sum().shift(1)
        n = t["n"].rolling(3, min_periods=3).sum().shift(1)
        out.append(pd.DataFrame({"cluster": c, "year": list(yrs), "lagshare": (s / n).where(n >= 10).values}))
    return d.merge(pd.concat(out), on=["cluster", "year"], how="left")


# ───────────────────────── estimation engine ─────────────────────────
def design(d, xvars, fe):
    parts = [d[xvars].astype(float)]
    for f in fe:
        parts.append(pd.get_dummies(d[f].astype(str), prefix="fe_" + f, drop_first=True, dtype=float))
    M = pd.concat(parts, axis=1)
    M.insert(0, "const", 1.0)
    keep = (M.std() > 0) | (M.columns == "const")
    return M.loc[:, keep]


def crv1(y, X, gi):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    G = gi.max() + 1
    Sg = np.zeros((G, k)); np.add.at(Sg, gi, X * u[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtXi @ (Sg.T @ Sg) @ XtXi
    return b, V, XtXi


def wcb_p(y, X, gi, j, tstat, B):
    """Restricted wild-cluster bootstrap (Rademacher) p-value for coefficient j."""
    n, k = X.shape; G = gi.max() + 1
    Xr = np.delete(X, j, axis=1)
    fr = Xr @ (np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)); ur = y - fr
    XtXi = np.linalg.pinv(X.T @ X); P = XtXi @ X.T
    xa = X @ XtXi[j]                      # so that V_jj = c * sum_g (sum_i xa_i u_i)^2
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    Gm = sp.csr_matrix((np.ones(n), (gi, np.arange(n))), shape=(G, n))
    cnt = 0
    for _ in range(B):
        w = RNG.choice([-1.0, 1.0], size=G)[gi]
        ys = fr + w * ur
        bs = P @ ys; us = ys - X @ bs
        sc = Gm @ (xa * us)
        t = bs[j] / np.sqrt(c * (sc @ sc))
        cnt += abs(t) >= abs(tstat)
    return (cnt + 1) / (B + 1)


def fit(d, y, xvars, fe=("cluster", "year"), boot=()):
    d = d.dropna(subset=[y] + xvars).copy()
    M = design(d, xvars, fe); names = list(M.columns)
    X = M.values; yv = d[y].values.astype(float)
    gi = pd.factorize(d["cluster"])[0]
    b, V, _ = crv1(yv, X, gi)
    G = gi.max() + 1; crit = ss.t.ppf(.975, G - 1)
    res = dict(b=b, V=V, names=names, G=G, n=len(d), coefs={})
    for t in xvars:
        if t not in names:
            continue
        j = names.index(t); se = np.sqrt(V[j, j]); tt = b[j] / se
        r = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se,
                 p=2 * ss.t.sf(abs(tt), G - 1), p_wcb=np.nan)
        if t in boot and B_BOOT > 0:
            r["p_wcb"] = wcb_p(yv, X, gi, j, tt, B_BOOT)
        res["coefs"][t] = r
    return res


def lincom(res, w):
    idx = [res["names"].index(k) for k in w]; wv = np.array(list(w.values()))
    est = wv @ res["b"][idx]; se = np.sqrt(wv @ res["V"][np.ix_(idx, idx)] @ wv)
    crit = ss.t.ppf(.975, res["G"] - 1)
    return dict(beta=est, se=se, lo=est - crit * se, hi=est + crit * se,
                p=2 * ss.t.sf(abs(est / se), res["G"] - 1), p_wcb=np.nan)


def record(section, grade, spec, res, terms, n=None):
    for t in terms:
        if t in res["coefs"]:
            ROWS.append(dict(section=section, grade=grade, spec=spec, term=t, n=res["n"], G=res["G"],
                             **res["coefs"][t]))


def record_lc(section, grade, spec, name, r, res):
    ROWS.append(dict(section=section, grade=grade, spec=spec, term=name, n=res["n"], G=res["G"], **r))


# ───────────────────────── formatting helpers ─────────────────────────
def pct(b): return (np.exp(b) - 1) * 100
def fp(p): return "n/a" if p is None or np.isnan(p) else ("<.001" if p < .001 else f"{p:.3f}")
def fr(r, boot=True):
    s = f"β={r['beta']:.3f}, 95% CI [{r['lo']:.3f}, {r['hi']:.3f}], cluster-robust p={fp(r['p'])}"
    if boot and not np.isnan(r.get("p_wcb", np.nan)):
        s += f", wild-bootstrap p={fp(r['p_wcb'])}"
    return s
def strength(r):
    p = r["p_wcb"] if not np.isnan(r.get("p_wcb", np.nan)) else r["p"]
    return "clear" if p < .05 else ("weak/marginal" if p < .10 else "not distinguishable from zero")


# ───────────────────────── analyses ─────────────────────────
def a0_descriptives(D):
    t = D.groupby("year").agg(n=("oa", "size"), oa_share=("oa", "mean"),
                              mean_lp_oa=("lp", lambda s: s[D.loc[s.index, "oa"] == 1].mean()),
                              mean_lp_closed=("lp", lambda s: s[D.loc[s.index, "oa"] == 0].mean()))
    t["raw_gap"] = t["mean_lp_oa"] - t["mean_lp_closed"]
    for c in ["gold", "green", "hybrid", "bronze"]:
        t["share_" + c] = D.groupby("year")["c_" + c].mean()
    t.to_csv(OUT / "T0_descriptives_by_year.csv")
    ct = D.groupby("colour").agg(n=("lp", "size"), mean_lp=("lp", "mean"), mean_lc=("lc", "mean"),
                                 mean_patents=("Citing_Patents", "mean"))
    ct.to_csv(OUT / "T0_descriptives_by_colour.csv")
    R["desc_colour"] = ct
    R["N_main"] = len(D); R["G"] = D["cluster"].nunique()
    R["oa_share"] = D["oa"].mean()


def a1_h1(D):
    sec = "H1"
    boot = ("oa",)
    specs = {
        "H1-a total association (no academic-citation control)": (D, ["oa"], ("cluster", "year")),
        "H1-b + log academic citations": (D, ["oa", "lc"], ("cluster", "year")),
        "H1-c + pub-type FE, field-tag count": (D, ["oa", "lc", "ln_fields"], ("cluster", "year", "pubtype")),
    }
    for k, (dd, xv, fe) in specs.items():
        res = fit(dd, "lp", xv, fe, boot=boot)
        record(sec, "ADJUSTED ASSOCIATION", k, res, ["oa"])
        R[k] = res["coefs"].get("oa")
    dd = D[D.colour.isin(["closed", "gold", "hybrid"])].copy(); dd["oa_pub"] = dd.colour.isin(["gold", "hybrid"]).astype(int)
    for nm, xv in [("H1-d Gold/Hybrid vs closed (open at publication), no lc", ["oa_pub"]),
                   ("H1-d' Gold/Hybrid vs closed, + lc", ["oa_pub", "lc"])]:
        res = fit(dd, "lp", xv, boot=("oa_pub",))
        record(sec, "ADJUSTED ASSOCIATION", nm, res, ["oa_pub"]); R[nm] = res["coefs"].get("oa_pub")
    dg = D[D.colour.isin(["closed", "green"])].copy(); dg["oa_gr"] = (dg.colour == "green").astype(int)
    res = fit(dg, "lp", ["oa_gr", "lc"], boot=("oa_gr",))
    record(sec, "ADJUSTED ASSOCIATION", "H1-e Green vs closed, + lc", res, ["oa_gr"]); R["H1-e"] = res["coefs"].get("oa_gr")
    try:
        M = design(D, ["oa", "lc"], ("cluster", "year"))
        gi = pd.factorize(D["cluster"])[0]
        m = sm.GLM(D["Citing_Patents"].astype(float).values, M.values, family=sm.families.Poisson()) \
            .fit(cov_type="cluster", cov_kwds={"groups": gi})
        j = list(M.columns).index("oa"); b, se = m.params[j], m.bse[j]
        r = dict(beta=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, p=m.pvalues[j], p_wcb=np.nan)
        record_lc(sec, "ADJUSTED ASSOCIATION", "H1-f PPML on patent count, + lc", "oa", r, dict(n=len(D), G=gi.max() + 1))
        R["H1-f"] = r
    except Exception as e:
        print("PPML failed:", e)


def a2_h2(D):
    sec = "H2"
    D = D.copy()
    lab = []
    for i, (a, b) in enumerate(PERIODS):
        D[f"oa_P{i}"] = D["oa"] * D["year"].between(a, b).astype(int); lab.append(f"{a}-{b}")
    pers = {}
    for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
        xv = [f"oa_P{i}" for i in range(len(PERIODS))] + extra
        res = fit(D, "lp", xv)
        record(sec, "DESCRIPTIVE / ADJUSTED", f"H2-a period-specific OA premium ({tag})", res, xv[:len(PERIODS)])
        first, last = f"oa_P0", f"oa_P{len(PERIODS)-1}"
        lc_ = lincom(res, {last: 1, first: -1})
        record_lc(sec, "DESCRIPTIVE / ADJUSTED", f"H2-a last minus first period ({tag})", "diff", lc_, res)
        pers[tag] = dict(res=res, diff=lc_)
    R["H2a"] = pers; R["period_labels"] = lab
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (tag, c) in enumerate([("no lc", "#1B5EA8"), ("+ lc", "#C0392B")]):
        rs = [pers[tag]["res"]["coefs"].get(f"oa_P{i}") for i in range(len(PERIODS))]
        x = np.arange(len(PERIODS)) + (k - .5) * .12
        ax.errorbar(x, [r["beta"] for r in rs],
                    yerr=[[r["beta"] - r["lo"] for r in rs], [r["hi"] - r["beta"] for r in rs]],
                    fmt="o-", capsize=4, color=c, label=f"OA premium ({tag})")
    ax.axhline(0, color="grey", ls=":"); ax.set_xticks(range(len(PERIODS))); ax.set_xticklabels(lab)
    ax.set_ylabel("OA − closed, log(1+patent citations)"); ax.legend(); ax.set_title("H2: OA premium by period (95% CI, t[G-1])")
    plt.tight_layout(); plt.savefig(OUT / "F_H2_period_premium.png", dpi=200); plt.close()
    sweep = []
    for c in BREAKS:
        D["oa_post"] = D["oa"] * (D["year"] >= c).astype(int)
        for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
            res = fit(D, "lp", ["oa", "oa_post"] + extra)
            record(sec, "DESCRIPTIVE / ADJUSTED", f"H2-b breakpoint {c} ({tag})", res, ["oa", "oa_post"])
            sweep.append((c, tag, res["coefs"]["oa_post"]))
    R["H2b"] = sweep
    D["lag_c"] = D["lagshare"] - D["lagshare"].mean()
    D["oa_x_lag"] = D["oa"] * D["lag_c"]
    for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
        res = fit(D, "lp", ["oa", "lag_c", "oa_x_lag"] + extra, boot=("oa_x_lag",))
        record(sec, "DESCRIPTIVE / ADJUSTED", f"H2-c OA x lagged cluster OA share ({tag})", res, ["oa", "oa_x_lag"])
        R[f"H2c {tag}"] = res["coefs"].get("oa_x_lag"); R[f"H2c_n {tag}"] = res["n"]
    D["oa_post15"] = D["oa"] * (D["year"] >= 2015).astype(int)
    res = fit(D, "lp", ["oa", "lag_c", "oa_x_lag", "oa_post15", "lc"])
    record(sec, "DESCRIPTIVE / ADJUSTED", "H2-d joint: OA x share and OA x post2015 (+lc)", res, ["oa_x_lag", "oa_post15"])
    R["H2d"] = res["coefs"]


def a3_h3(D):
    sec = "H3"
    cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"] + (["c_other"] if D["c_other"].sum() > 0 else [])
    M = {"M0 colour only": (cols, ("cluster", "year")),
         "M1 + log academic citations": (cols + ["lc"], ("cluster", "year")),
         "M2 + pub-type FE, field-tag count": (cols + ["lc", "ln_fields"], ("cluster", "year", "pubtype"))}
    R["H3"] = {}
    for k, (xv, fe) in M.items():
        res = fit(D, "lp", xv, fe, boot=("c_green", "c_gold"))
        record(sec, "ADJUSTED ASSOCIATION", k, res, cols)
        gg = lincom(res, {"c_green": 1, "c_gold": -1})
        record_lc(sec, "ADJUSTED ASSOCIATION", k + " | Green minus Gold", "green_minus_gold", gg, res)
        R["H3"][k] = dict(res=res, gg=gg)
    g0 = R["H3"]["M0 colour only"]; g1 = R["H3"]["M1 + log academic citations"]
    R["shrink_green"] = 1 - g1["res"]["coefs"]["c_green"]["beta"] / g0["res"]["coefs"]["c_green"]["beta"]
    R["shrink_gap"] = 1 - g1["gg"]["beta"] / g0["gg"]["beta"] if g0["gg"]["beta"] != 0 else np.nan
    for nm, dd in [("2003-2014", D[D.year <= 2014]), ("2015-2024", D[D.year >= 2015])]:
        try:
            res = fit(dd, "lp", cols + ["lc"])
            gg = lincom(res, {"c_green": 1, "c_gold": -1})
            record_lc(sec, "ADJUSTED ASSOCIATION", f"M1 Green minus Gold, {nm}", "green_minus_gold", gg, res)
            R.setdefault("H3split", {})[nm] = gg
        except Exception as e:
            print("split failed", nm, e)
    try:
        thr = D["Citing_Patents"].quantile(.99); D = D.assign(top1=(D["Citing_Patents"] >= thr).astype(int))
        Mx = design(D, cols + ["lc"], ("year",)); gi = pd.factorize(D["cluster"])[0]
        m = sm.Logit(D["top1"].values, Mx.values).fit(disp=0, cov_type="cluster", cov_kwds={"groups": gi})
        for c in cols:
            j = list(Mx.columns).index(c)
            r = dict(beta=m.params[j], se=m.bse[j], lo=m.params[j] - 1.96 * m.bse[j],
                     hi=m.params[j] + 1.96 * m.bse[j], p=m.pvalues[j], p_wcb=np.nan)
            record_lc(sec, "ADJUSTED ASSOCIATION", "top-1% logit (log-odds), + lc, year FE", c, r, dict(n=len(D), G=gi.max() + 1))
    except Exception as e:
        print("logit failed:", e)
    # NOTE: c_other (65 records, no top-1% papers) produced quasi-complete separation (beta ~ -14.8, p ~ 1e-210): an artefact.
    gdf = D[D.colour == "green"]
    diag = {"n_green": len(gdf)}
    if "PMCID" in D.columns:
        diag["share_green_with_PMCID"] = gdf["PMCID"].notna().mean()
    if "Open Access License" in D.columns:
        diag["share_green_with_license_field"] = gdf["Open Access License"].notna().mean()
    R["green_diag"] = diag
    pd.Series(diag).to_csv(OUT / "T_H3_green_version_diagnostic.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (nm, c) in enumerate([("M0 colour only", "#7F8C8D"), ("M1 + log academic citations", "#1B5EA8")]):
        rs = [R["H3"][nm]["res"]["coefs"][x] for x in cols]
        x = np.arange(len(cols)) + (k - .5) * .15
        ax.errorbar(x, [r["beta"] for r in rs], yerr=[[r["beta"] - r["lo"] for r in rs], [r["hi"] - r["beta"] for r in rs]],
                    fmt="o", capsize=4, color=c, label=nm)
    ax.axhline(0, color="grey", ls=":"); ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2:] for c in cols])
    ax.set_ylabel("vs closed, log(1+patent citations)"); ax.legend(fontsize=8)
    ax.set_title("H3: colour associations before/after academic-citation adjustment")
    plt.tight_layout(); plt.savefig(OUT / "F_H3_colour.png", dpi=200); plt.close()


def _dm(d, xv):
    return design(d, xv, ("cluster", "year"))


def a4_decomposition(D):
    """Baron-Kenny-style statistical decomposition; cluster bootstrap CI. NOT a mechanism test."""
    def est(d):
        Xt = _dm(d, ["oa"]); Xd = _dm(d, ["oa", "lc"])
        tot = np.linalg.lstsq(Xt.values, d["lp"].values, rcond=None)[0][list(Xt.columns).index("oa")]
        bd = np.linalg.lstsq(Xd.values, d["lp"].values, rcond=None)[0]
        dire = bd[list(Xd.columns).index("oa")]; bl = bd[list(Xd.columns).index("lc")]
        a = np.linalg.lstsq(Xt.values, d["lc"].values, rcond=None)[0][list(Xt.columns).index("oa")]
        return tot, dire, a, bl, tot - dire
    tot, dire, a, bl, ind = est(D)
    cl = D["cluster"].unique(); bs = []
    for _ in range(B_MED):
        pick = RNG.choice(cl, size=len(cl), replace=True)
        parts = []
        for i, c in enumerate(pick):
            g = D[D.cluster == c].copy(); g["cluster"] = f"b{i}"; parts.append(g)
        try:
            t2, d2, _, _, i2 = est(pd.concat(parts)); bs.append((t2, d2, i2, i2 / t2 if t2 else np.nan))
        except Exception:
            continue
    bs = np.array(bs)
    ci = lambda k: (np.nanpercentile(bs[:, k], 2.5), np.nanpercentile(bs[:, k], 97.5))
    R["decomp"] = dict(total=tot, direct=dire, indirect=ind, share=ind / tot, a=a, b=bl,
                       ci_total=ci(0), ci_direct=ci(1), ci_ind=ci(2), ci_share=ci(3))
    pd.DataFrame([R["decomp"]]).to_csv(OUT / "T_decomposition.csv", index=False)


def a5_window(D_all):
    rows = []
    for a, b in [(YMIN, YMAX), (1992, 2024), (1992, 2025), (YMIN, 2025), (2010, 2024)]:
        dd = D_all[D_all.year.between(a, b)]
        if len(dd) < 500:
            continue
        r1 = fit(dd, "lp", ["oa", "lc"])["coefs"]["oa"]
        dg = dd[dd.colour.isin(["closed", "gold", "hybrid"])].copy(); dg["oa_pub"] = dg.colour.isin(["gold", "hybrid"]).astype(int)
        r2 = fit(dg, "lp", ["oa_pub", "lc"])["coefs"]["oa_pub"]
        rows.append(dict(window=f"{a}-{b}", n=len(dd), oa_beta=r1["beta"], oa_lo=r1["lo"], oa_hi=r1["hi"], oa_p=r1["p"],
                         goldhybrid_beta=r2["beta"], goldhybrid_lo=r2["lo"], goldhybrid_hi=r2["hi"], goldhybrid_p=r2["p"]))
    t = pd.DataFrame(rows); t.to_csv(OUT / "T_sample_window_sensitivity.csv", index=False); R["window"] = t


# ───────────────────────── report ─────────────────────────
def write_report():
    L = []; w = L.append
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    w(f"# Redesign blueprint (auto-generated {now})\n")
    w(f"Main sample: {YMIN}-{YMAX}, N={R['N_main']:,}, clusters G={R['G']}, OA share={R['oa_share']:.1%}. "
      f"All numbers below are recomputed from the current data.\n")
    a = R["H1-a total association (no academic-citation control)"]; b = R["H1-b + log academic citations"]
    c = R["H1-c + pub-type FE, field-tag count"]
    w(f"**C1 (H1).** OA-classified papers: {pct(b['beta']):.1f}% higher expected log-patent intensity conditional on cluster, year and academic citations ({fr(b)}; evidence {strength(b)}). "
      f"Total association {pct(a['beta']):.1f}% ({fr(a)}). With pub-type FE and field tags {pct(c['beta']):.1f}% ({fr(c)}).\n")
    d0 = R["H1-d Gold/Hybrid vs closed (open at publication), no lc"]; d1 = R["H1-d' Gold/Hybrid vs closed, + lc"]
    w(f"**C2 (timing).** Gold/Hybrid vs closed: {pct(d0['beta']):.1f}% without and {pct(d1['beta']):.1f}% with academic citations ({fr(d1)}; evidence {strength(d1)}).\n")
    if "H1-f" in R:
        w(f"PPML on counts: {pct(R['H1-f']['beta']):.1f}% ({fr(R['H1-f'], boot=False)}).\n")
    h = R["H2a"]
    for tag in ["no lc", "+ lc"]:
        dd = h[tag]["diff"]
        w(f"**C3 (H2, {tag}).** Last-period minus first-period OA premium ({R['period_labels'][-1]} vs {R['period_labels'][0]}): {dd['beta']:.3f} ({fr(dd, boot=False)}).\n")
    w("Breakpoint sweep (OA x Post):\n")
    w("| cut-off | spec | OA x Post β | p |\n|---|---|---|---|")
    for cyr, tag, r in R["H2b"]:
        w(f"| {cyr} | {tag} | {r['beta']:.3f} | {fp(r['p'])} |")
    w("")
    m1 = R["H2c no lc"]; m2 = R["H2c + lc"]
    if m1 is not None:
        w(f"**C4 (moderation by lagged cluster OA share; n={R['H2c_n no lc']:,}).** OA x lagged-share β={m1['beta']:.3f} ({fr(m1)}) / {m2['beta']:.3f} ({fr(m2)}).\n")
    jd = R["H2d"]
    if "oa_x_lag" in jd and "oa_post15" in jd:
        w(f"Joint: OA x share β={jd['oa_x_lag']['beta']:.3f} (p={fp(jd['oa_x_lag']['p'])}); OA x post-2015 β={jd['oa_post15']['beta']:.3f} (p={fp(jd['oa_post15']['p'])}).\n")
    H3 = R["H3"]; g0 = H3["M0 colour only"]; g1 = H3["M1 + log academic citations"]; g2 = H3["M2 + pub-type FE, field-tag count"]
    w(f"**C6 (H3).** Green minus Gold: M0 {g0['gg']['beta']:.3f} ({fr(g0['gg'], boot=False)}); M1 {g1['gg']['beta']:.3f} ({fr(g1['gg'], boot=False)}); "
      f"M2 {g2['gg']['beta']:.3f} ({fr(g2['gg'], boot=False)}). Green coefficient shrinks {R['shrink_green']*100:.0f}%, the gap {R['shrink_gap']*100:.0f}% (M0 to M1).\n")
    sp_ = R.get("H3split", {})
    if sp_:
        w("Green-Gold gap by period (M1): " + "; ".join(f"{k}: {v['beta']:.3f} (p={fp(v['p'])})" for k, v in sp_.items()) + ".\n")
    w(f"Green version diagnostic: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in R['green_diag'].items()})}.\n")
    dc = R["decomp"]
    w(f"**C7 (appendix, statistical decomposition).** Total {dc['total']:.3f} [{dc['ci_total'][0]:.3f}, {dc['ci_total'][1]:.3f}]; direct {dc['direct']:.3f}; "
      f"via academic citations {dc['indirect']:.3f} [{dc['ci_ind'][0]:.3f}, {dc['ci_ind'][1]:.3f}] ({dc['share']*100:.0f}% of total).\n")
    w("## Sample-window sensitivity\n"); w(R["window"].round(4).to_markdown(index=False)); w("")
    (OUT / "REDESIGN_BLUEPRINT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    D_all = add_lag_share(load())
    D = D_all[D_all.year.between(YMIN, YMAX)].copy()
    print(f"main N={len(D):,}, clusters={D.cluster.nunique()}, colours={D.colour.value_counts().to_dict()}")
    a0_descriptives(D); a1_h1(D); a2_h2(D); a3_h3(D); a4_decomposition(D); a5_window(D_all)
    pd.DataFrame(ROWS).to_csv(OUT / "ALL_COEFFICIENTS.csv", index=False)
    write_report()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
