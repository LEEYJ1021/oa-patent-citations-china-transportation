#!/usr/bin/env python3
"""
STEP 3 - PRE-SPECIFIED PRIMARY ANALYSIS (zero-truncated count models)
=====================================================================
Design principle: the specification is written to PRIMARY_ANALYSIS_PLAN.md (with timestamp and
SHA-256 of the spec) BEFORE any data are read or model is fitted. Every fitted coefficient is
written to the ledger; nothing is filtered by significance.

Why zero-truncated: the sample is restricted to papers with >=1 citing patent, so the correct
likelihood for Citing_Patents is P(y | y>=1). Log-OLS and untruncated PPML are sensitivity only.
Note on exposure: cluster + year fixed effects already absorb paper age (age is a function of
year), so an offset(log age) would be redundant and is deliberately NOT used. The OA rate
ratio is identified from OA vs closed papers of the same year and cluster.

Input : Transport_CN_Scholarly_Works_with_cluster.csv  (same as previous steps)
Env   : RV_RAW, RV_OUT (./outputs/step3), RV_YEAR_MIN (2003), RV_YEAR_MAX (2024), RV_BOOT (999)
Optional columns (used automatically if present):
  first_oa_year  -> open-at-publication OA indicator (R2-4/R2-5)
  zero-citation rows (Citing_Patents==0) -> hurdle part (any patent citation) is added
Run   : RV_RAW=./outputs/Transport_CN_Scholarly_Works_with_cluster.csv python3 -u step3_primary_plan.py
Requires: pandas numpy scipy statsmodels tabulate
"""
import os, re, json, hashlib, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.sparse as sp
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

warnings.filterwarnings("ignore")
RAW = Path(os.environ.get("RV_RAW", "./outputs/Transport_CN_Scholarly_Works_with_cluster.csv"))
OUT = Path(os.environ.get("RV_OUT", "./outputs/step3")); OUT.mkdir(parents=True, exist_ok=True)
YMIN = int(os.environ.get("RV_YEAR_MIN", 2003)); YMAX = int(os.environ.get("RV_YEAR_MAX", 2024))
B_BOOT = int(os.environ.get("RV_BOOT", 999))
RNG = np.random.default_rng(2026)
PERIODS = [(2003, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]

# ───────────────────────── 0. PLAN (fixed before estimation) ─────────────────────────
PLAN = {
    "sample": f"{YMIN}-{YMAX}; Citing_Patents>=1 (intensive margin); 25 clusters; SE = CRV1 with t(G-1)",
    "model_primary": "zero-truncated Poisson (ZTP), cluster FE + year FE; effects reported as rate ratios exp(beta)",
    "model_sensitivity": "zero-truncated NegBin (ZTNB); log(1+y) OLS and PPML appear only in earlier steps",
    "P1_H1_total": "ZTP: patents ~ OA  (no academic-citation control)",
    "P2_H1_adj": "ZTP: patents ~ OA + log(1+academic citations)",
    "P3_H2_contrast": "ZTP: OA premium in 2015-19 minus 2003-09 (both periods >=5y exposure), no academic-citation control; "
                      "estimated by reparametrisation so the contrast has its own wild-score-bootstrap p",
    "P4_H3_gap": "ZTP: Green minus Gold, colour dummies + log academic citations (closed = reference, 'other' colour dropped)",
    "primary_family": ["P1_H1_total", "P2_H1_adj", "P3_H2_contrast", "P4_H3_gap"],
    "multiplicity": "Holm correction over the four primary tests (cluster-robust t(G-1) p-values)",
    "secondary": "with/without academic citations for every block; pub-type FE; open-at-publication (Gold/Hybrid vs closed); "
                 "pub<=2019 subsample; OA x Post 2013/2015; OA x lagged cluster OA share (exploratory); ZTNB; hurdle if zeros exist",
    "reporting_rules": [
        "All coefficients in the ledger are reported, whatever their sign or significance.",
        "Hypotheses H2/H3 were re-formulated after review and after first-round results; label them exploratory.",
        "A non-significant contrast is described as 'not distinguishable from zero' with its CI, never as 'no effect'.",
        "Cluster-level OA share and OA status are retrospective (measured at extraction); state this as a limitation.",
    ],
}
PLAN_HASH = hashlib.sha256(json.dumps(PLAN, sort_keys=True).encode()).hexdigest()


def write_plan():
    L = [f"# Primary analysis plan (written {datetime.now():%Y-%m-%d %H:%M:%S}, before estimation)",
         f"SHA-256 of spec: `{PLAN_HASH}`\n"]
    for k, v in PLAN.items():
        if isinstance(v, list):
            L.append(f"**{k}**"); L += [f"- {x}" for x in v]
        else:
            L.append(f"- **{k}**: {v}")
    (OUT / "PRIMARY_ANALYSIS_PLAN.md").write_text("\n".join(L), encoding="utf-8")


# ───────────────────────── data ─────────────────────────
def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "MacRoman", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("cannot read " + str(path))


def load():
    d = read_csv(RAW); d.columns = d.columns.str.strip()
    miss = [c for c in ["Pub_Year", "Citing_Patents", "Citing_Works", "cluster"] if c not in d.columns]
    if miss:
        raise SystemExit(f"missing columns {miss}")
    if "OA" not in d.columns:
        d["OA"] = d["Is Open Access"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["oa"] = d["OA"].astype(int)
    d["year"] = pd.to_numeric(d["Pub_Year"], errors="coerce")
    d = d.dropna(subset=["year", "Citing_Patents", "cluster"]).copy()
    d["year"] = d["year"].astype(int)
    d["y"] = d["Citing_Patents"].astype(float)
    d["lc"] = np.log1p(pd.to_numeric(d["Citing_Works"], errors="coerce").fillna(0))
    col = d["Open Access Colour"].fillna("").astype(str).str.strip().str.lower() \
        if "Open Access Colour" in d.columns else pd.Series("", index=d.index)
    d["colour"] = np.where(d["oa"] == 0, "closed",
                           np.where(col.isin(["gold", "green", "hybrid", "bronze"]), col, "other"))
    for c in ["gold", "green", "hybrid", "bronze"]:
        d["c_" + c] = (d["colour"] == c).astype(int)
    pt = d["Pub_Type"] if "Pub_Type" in d.columns else d.get("Publication Type", pd.Series("unk", index=d.index))
    d["pubtype"] = pt.fillna("unk").astype(str).str.lower()
    fos = d["Fields of Study"].fillna("") if "Fields of Study" in d.columns else pd.Series("", index=d.index)
    d["ln_fields"] = np.log1p(fos.apply(lambda s: len([x for x in re.split(r"[;,]", s) if x.strip()])))
    d["cluster"] = d["cluster"].astype(int).astype(str)
    return d.reset_index(drop=True)


def add_lag_share(d):
    out = []; yrs = range(int(d.year.min()), int(d.year.max()) + 1)
    for c, g in d.groupby("cluster"):
        t = g.groupby("year")["oa"].agg(s="sum", n="count").reindex(yrs, fill_value=0)
        s = t["s"].rolling(3, min_periods=3).sum().shift(1); n = t["n"].rolling(3, min_periods=3).sum().shift(1)
        out.append(pd.DataFrame({"cluster": c, "year": list(yrs), "lagshare": (s / n).where(n >= 10).values}))
    return d.merge(pd.concat(out), on=["cluster", "year"], how="left")


def design(d, xvars, fe):
    parts = [d[xvars].astype(float)]
    for f in fe:
        parts.append(pd.get_dummies(d[f].astype(str), prefix="fe_" + f, drop_first=True, dtype=float))
    M = pd.concat(parts, axis=1); M.insert(0, "const", 1.0)
    keep = (M.std() > 0) | (M.columns == "const")
    return M.loc[:, keep]


# ───────────────────────── zero-truncated Poisson ─────────────────────────
def _sw(X, y, b):
    mu = np.exp(np.clip(X @ b, -20, 20)); q = -np.expm1(-mu)
    s = y - mu / q                                   # d loglik / d eta
    W = mu * (q - mu * np.exp(-mu)) / q ** 2         # -d2 loglik / d eta2  (>0)
    return s, W, mu


def _ll(X, y, b):
    mu = np.exp(np.clip(X @ b, -20, 20))
    return (y * np.log(mu) - mu - gammaln(y + 1) - np.log(-np.expm1(-mu))).sum()


def _newton(X, y, b, it=60):
    for _ in range(it):
        s, W, _ = _sw(X, y, b)
        step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-9 * np.eye(X.shape[1]), X.T @ s)
        l0, t = _ll(X, y, b), 1.0
        while t > 1e-4 and _ll(X, y, b + t * step) < l0:
            t /= 2
        b = b + t * step
        if np.max(np.abs(t * step)) < 1e-9:
            break
    return b


def _start(X, y):
    return np.r_[np.log(max(y.mean() - .5, .3)), np.zeros(X.shape[1] - 1)]


def ztp_fit(d, xvars, fe=("cluster", "year"), boot=()):
    d = d[d["y"] >= 1].dropna(subset=xvars).copy()
    M = design(d, xvars, fe); names = list(M.columns); X = M.values; y = d["y"].values
    gi = pd.factorize(d["cluster"])[0]; G = gi.max() + 1; n, k = X.shape
    b = _newton(X, y, _start(X, y))
    s, W, _ = _sw(X, y, b)
    Hinv = np.linalg.pinv(X.T @ (W[:, None] * X))
    Sg = np.zeros((G, k)); np.add.at(Sg, gi, X * s[:, None])
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * Hinv @ (Sg.T @ Sg) @ Hinv
    crit = ss.t.ppf(.975, G - 1)
    res = dict(b=b, V=V, names=names, G=G, n=n, coefs={})
    for t_ in xvars:
        if t_ not in names:
            continue
        j = names.index(t_); se = np.sqrt(V[j, j]); tt = b[j] / se
        r = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se,
                 p=2 * ss.t.sf(abs(tt), G - 1), p_wcb=np.nan)
        if t_ in boot and B_BOOT > 0:
            r["p_wcb"] = _wild_score_p(X, y, gi, j, tt, b, c)
        res["coefs"][t_] = r
    return res


def _wild_score_p(X, y, gi, j, tobs, b_full, c):
    """Null-imposed wild score bootstrap (Rademacher, Kline-Santos) p-value for coefficient j."""
    n, k = X.shape; G = gi.max() + 1
    Xr = np.delete(X, j, 1)
    br = _newton(Xr, y, np.delete(b_full, j)); b0 = np.insert(br, j, 0.0)
    s, W, _ = _sw(X, y, b0)
    Hinv = np.linalg.pinv(X.T @ (W[:, None] * X)); a = Hinv[j]
    Sg = np.zeros((G, k)); np.add.at(Sg, gi, X * s[:, None])
    Ag = np.zeros((G, k))
    for g in range(G):
        m = gi == g; Ag[g] = a @ (X[m].T @ (W[m][:, None] * X[m]))
    sa = Sg @ a; cnt = 0
    for _ in range(B_BOOT):
        w = RNG.choice([-1.0, 1.0], size=G)
        Dl = Hinv @ (Sg.T @ w)
        q = w * sa - Ag @ Dl
        cnt += abs(Dl[j] / np.sqrt(c * (q @ q))) >= abs(tobs)
    return (cnt + 1) / (B_BOOT + 1)


# ───────────────────────── zero-truncated NegBin (sensitivity) ─────────────────────────
class ZTNB(GenericLikelihoodModel):
    def loglikeobs(self, params):
        r = 1 / np.exp(np.clip(params[-1], -8, 5)); y = self.endog
        mu = np.exp(np.clip(self.exog @ params[:-1], -20, 20)); lr = np.log(r) - np.log(r + mu)
        return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * lr + y * (np.log(mu) - np.log(r + mu))
                - np.log(-np.expm1(r * lr)))


def ztnb_fit(d, xvars, fe=("cluster", "year")):
    # NOTE: this statsmodels-based estimator failed to converge on the real data;
    # step5_full_pipeline.py replaces it (s3.ztnb_fit = ztnb_fit) with an analytic-score estimator.
    d = d[d["y"] >= 1].dropna(subset=xvars).copy()
    M = design(d, xvars, fe); names = list(M.columns); X = M.values; y = d["y"].values
    gi = pd.factorize(d["cluster"])[0]; G = gi.max() + 1
    st = np.r_[_newton(X, y, _start(X, y)), np.log(.3)]
    m = ZTNB(y, X).fit(start_params=st, method="bfgs", maxiter=3000, disp=0,
                       cov_type="cluster", cov_kwds={"groups": gi})
    b = np.asarray(m.params); V = np.asarray(m.cov_params()); crit = ss.t.ppf(.975, G - 1)
    coefs = {}
    for t_ in xvars:
        if t_ in names:
            j = names.index(t_); se = np.sqrt(V[j, j])
            coefs[t_] = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se,
                             p=2 * ss.t.sf(abs(b[j] / se), G - 1), p_wcb=np.nan)
    return dict(b=b[:-1], V=V[:-1, :-1], names=names, G=G, n=len(d), coefs=coefs)


def lincom(res, w):
    idx = [res["names"].index(k) for k in w]; wv = np.array(list(w.values()))
    est = wv @ res["b"][idx]; se = np.sqrt(wv @ res["V"][np.ix_(idx, idx)] @ wv); crit = ss.t.ppf(.975, res["G"] - 1)
    return dict(beta=est, se=se, lo=est - crit * se, hi=est + crit * se,
                p=2 * ss.t.sf(abs(est / se), res["G"] - 1), p_wcb=np.nan)


# ───────────────────────── ledger ─────────────────────────
ROWS = []; PRIM = {}


def rec(block, spec, term, r, res, primary=None):
    ROWS.append(dict(block=block, spec=spec, term=term, n=res["n"], G=res["G"],
                     RR=np.exp(r["beta"]), RR_lo=np.exp(r["lo"]), RR_hi=np.exp(r["hi"]), **r))
    if primary:
        PRIM[primary] = dict(spec=spec, term=term, **r)


def run(D, D_all):
    # ---- P1/P2 + H1 secondary ----
    for tag, xv, fe, prim in [("H1 total (no lc)", ["oa"], ("cluster", "year"), "P1_H1_total"),
                              ("H1 + lc", ["oa", "lc"], ("cluster", "year"), "P2_H1_adj"),
                              ("H1 + lc + pub-type FE + field tags", ["oa", "lc", "ln_fields"], ("cluster", "year", "pubtype"), None)]:
        res = ztp_fit(D, xv, fe, boot=("oa",)); rec("H1", "ZTP " + tag, "oa", res["coefs"]["oa"], res, prim)
    for tag, xv in [("H1 ZTNB total", ["oa"]), ("H1 ZTNB + lc", ["oa", "lc"])]:
        try:
            res = ztnb_fit(D, xv); rec("H1", tag, "oa", res["coefs"]["oa"], res)
        except Exception as e:
            print("ZTNB failed:", tag, e)
    dgh = D[D.colour.isin(["closed", "gold", "hybrid"])].copy(); dgh["oa_pub"] = dgh.colour.isin(["gold", "hybrid"]).astype(int)
    for tag, xv in [("H1 Gold/Hybrid vs closed (open at pub), no lc", ["oa_pub"]), ("H1 Gold/Hybrid vs closed, + lc", ["oa_pub", "lc"])]:
        res = ztp_fit(dgh, xv, boot=("oa_pub",)); rec("H1", "ZTP " + tag, "oa_pub", res["coefs"]["oa_pub"], res)
    if "first_oa_year" in D.columns:
        dd = D.copy(); f = pd.to_numeric(dd["first_oa_year"], errors="coerce")
        dd["oa_atpub"] = ((f <= dd["year"]) & (dd["oa"] == 1)).astype(int)
        dd = dd[(dd["oa"] == 0) | f.notna()]
        for tag, xv in [("H1 OA at publication (first_oa_year<=year), no lc", ["oa_atpub"]), ("... + lc", ["oa_atpub", "lc"])]:
            res = ztp_fit(dd, xv, boot=("oa_atpub",)); rec("H1", "ZTP " + tag, "oa_atpub", res["coefs"]["oa_atpub"], res)
    # hurdle part, only if zero-citation papers exist
    if (D_all["y"] == 0).any():
        dz = D_all[D_all.year.between(YMIN, YMAX)].copy(); dz["any"] = (dz["y"] >= 1).astype(float)
        M = design(dz, ["oa", "lc"], ("cluster", "year")); gi = pd.factorize(dz["cluster"])[0]
        m = sm.Logit(dz["any"].values, M.values).fit(disp=0, cov_type="cluster", cov_kwds={"groups": gi})
        j = list(M.columns).index("oa"); crit = ss.t.ppf(.975, gi.max())
        r = dict(beta=m.params[j], se=m.bse[j], lo=m.params[j] - crit * m.bse[j], hi=m.params[j] + crit * m.bse[j],
                 p=2 * ss.t.sf(abs(m.params[j] / m.bse[j]), gi.max()), p_wcb=np.nan)
        rec("H1", "HURDLE part 1: logit(any patent citation) (beta = log-odds)", "oa", r, dict(n=len(dz), G=gi.max() + 1))

    # ---- P3: H2 ----
    def per_cols(d):
        d = d.copy()
        for i, (a, b) in enumerate(PERIODS):
            d[f"oa_P{i}"] = d["oa"] * d["year"].between(a, b).astype(int)
        d["oa_P02"] = d["oa_P0"] + d["oa_P2"]; d["oa_negP0"] = -d["oa_P0"]   # theta = premium(P2) - premium(P0)
        return d
    Dp = per_cols(D)
    for tag, extra, prim in [("no lc", [], "P3_H2_contrast"), ("+ lc", ["lc"], None)]:
        res = ztp_fit(Dp, [f"oa_P{i}" for i in range(4)] + extra)
        for i in range(4):
            if f"oa_P{i}" in res["coefs"]:
                rec("H2", f"ZTP period premium {PERIODS[i][0]}-{PERIODS[i][1]} ({tag})", f"oa_P{i}", res["coefs"][f"oa_P{i}"], res)
        if "oa_P3" in res["coefs"]:
            rec("H2", f"ZTP last minus first ({tag})", "contrast", lincom(res, {"oa_P3": 1, "oa_P0": -1}), res)
        res = ztp_fit(Dp, ["oa_P1", "oa_P3", "oa_P02", "oa_negP0"] + extra, boot=("oa_negP0",))
        rec("H2", f"ZTP 2015-19 minus 2003-09 ({tag}) [reparametrised]", "oa_negP0", res["coefs"]["oa_negP0"], res, prim)
    Dp19 = Dp[Dp.year <= 2019]
    for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
        res = ztp_fit(Dp19, ["oa_P1", "oa_P02", "oa_negP0"] + extra, boot=("oa_negP0",))
        rec("H2", f"ZTP 2015-19 minus 2003-09, pub<=2019 ({tag})", "oa_negP0", res["coefs"]["oa_negP0"], res)
    dgh = Dp[Dp.colour.isin(["closed", "gold", "hybrid"])].copy(); dgh["oa"] = dgh.colour.isin(["gold", "hybrid"]).astype(int)
    dgh = per_cols(dgh)
    res = ztp_fit(dgh, ["oa_P1", "oa_P3", "oa_P02", "oa_negP0"], boot=("oa_negP0",))
    rec("H2", "ZTP 2015-19 minus 2003-09, Gold/Hybrid vs closed (no lc)", "oa_negP0", res["coefs"]["oa_negP0"], res)
    for cut in (2013, 2015):
        d = D.copy(); d["oa_post"] = d["oa"] * (d["year"] >= cut).astype(int)
        for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
            res = ztp_fit(d, ["oa", "oa_post"] + extra, boot=("oa_post",))
            rec("H2", f"ZTP OA x Post{cut} ({tag}) [secondary; period includes 2020-24]", "oa_post", res["coefs"]["oa_post"], res)
    d = D.dropna(subset=["lagshare"]).copy(); d["lag_c"] = d["lagshare"] - d["lagshare"].mean(); d["oa_x_lag"] = d["oa"] * d["lag_c"]
    for tag, extra in [("no lc", []), ("+ lc", ["lc"])]:
        res = ztp_fit(d, ["oa", "lag_c", "oa_x_lag"] + extra, boot=("oa_x_lag",))
        rec("H2", f"ZTP OA x lagged cluster OA share ({tag}) [exploratory]", "oa_x_lag", res["coefs"]["oa_x_lag"], res)

    # ---- P4: H3 ----
    d = D[D.colour != "other"].copy(); cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"]
    for tag, extra, fe, prim in [("M0 colour only", [], ("cluster", "year"), None),
                                 ("M1 + lc", ["lc"], ("cluster", "year"), "P4_H3_gap"),
                                 ("M2 + lc + pub-type FE + field tags", ["lc", "ln_fields"], ("cluster", "year", "pubtype"), None)]:
        res = ztp_fit(d, cols + extra, fe, boot=("c_green", "c_gold"))
        for c_ in cols:
            rec("H3", f"ZTP {tag}", c_, res["coefs"][c_], res)
        rec("H3", f"ZTP {tag} - Green minus Gold", "green_minus_gold", lincom(res, {"c_green": 1, "c_gold": -1}), res, prim)
    for nm, dd in [("2003-2014", d[d.year <= 2014]), ("2015-2024", d[d.year >= 2015])]:
        res = ztp_fit(dd, cols + ["lc"])
        rec("H3", f"ZTP M1 Green minus Gold, {nm}", "green_minus_gold", lincom(res, {"c_green": 1, "c_gold": -1}), res)
    try:
        res = ztnb_fit(d, cols + ["lc"])
        rec("H3", "ZTNB M1 Green minus Gold", "green_minus_gold", lincom(res, {"c_green": 1, "c_gold": -1}), res)
    except Exception as e:
        print("ZTNB H3 failed:", e)


# ───────────────────────── report ─────────────────────────
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run_ = 0
    for r, i in enumerate(o):
        run_ = max(run_, (m - r) * p[i]); adj[i] = min(1, run_)
    return adj


def fp(p): return "n/a" if pd.isna(p) else ("<.001" if p < .001 else f"{p:.3f}")


def report():
    L = []; w = L.append
    w(f"# STEP 3 report (plan hash `{PLAN_HASH[:16]}...`; sample {YMIN}-{YMAX})\n")
    keys = [k for k in PLAN["primary_family"] if k in PRIM]
    ph = holm(np.array([PRIM[k]["p"] for k in keys]))
    w("## Primary family (Holm-adjusted over the pre-specified four tests)\n")
    w("| test | spec | RR | 95% CI | p (t[G-1]) | p (wild score bootstrap) | Holm p |\n|---|---|---|---|---|---|---|")
    for k, a in zip(keys, ph):
        r = PRIM[k]
        w(f"| {k} | {r['spec']} | {np.exp(r['beta']):.3f} | [{np.exp(r['lo']):.3f}, {np.exp(r['hi']):.3f}] | "
          f"{fp(r['p'])} | {fp(r['p_wcb'])} | {fp(a)} |")
    w("\nRR = rate ratio (multiplicative effect on expected patent citations, conditional on >=1). "
      "For contrasts, RR is the ratio of premiums (P3: premium 2015-19 / premium 2003-09; P4: Green / Gold).\n")
    L2 = pd.DataFrame(ROWS)
    show = L2[["block", "spec", "term", "n", "RR", "RR_lo", "RR_hi", "p", "p_wcb"]].copy()
    for c in ("RR", "RR_lo", "RR_hi"): show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["p"] = show["p"].map(fp); show["p_wcb"] = show["p_wcb"].map(fp)
    for blk in ("H1", "H2", "H3"):
        w(f"## {blk} - all fitted coefficients\n"); w(show[show.block == blk].drop(columns="block").to_markdown(index=False)); w("")
    w("## Reading rules (from the plan)\n" + "\n".join("- " + x for x in PLAN["reporting_rules"]))
    w("- P3/H2: RR<1 with a CI containing 1 => 'premium not distinguishable between periods', not 'no compression'.")
    w("- If ZTP and ZTNB disagree materially, report both; ZTP standard errors are cluster-robust, so overdispersion mainly affects efficiency.")
    (OUT / "STEP3_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    write_plan()                                            # BEFORE any estimation
    D_all = add_lag_share(load())
    D = D_all[D_all.year.between(YMIN, YMAX) & (D_all["y"] >= 1)].copy()
    print(f"plan hash {PLAN_HASH[:12]} | N(y>=1)={len(D):,} clusters={D.cluster.nunique()}")
    run(D, D_all)
    pd.DataFrame(ROWS).to_csv(OUT / "S3_ALL_COEFFICIENTS.csv", index=False)
    report(); print("done ->", OUT)


if __name__ == "__main__":
    main()
