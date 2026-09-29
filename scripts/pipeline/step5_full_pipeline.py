#!/usr/bin/env python3
"""
STEP 5 (REVISED) - FULL PIPELINE
================================
Needs in the same folder: step3_primary_plan.py, step4_revised_H2H3.py

Changes vs. the previous Step 5
  * REMOVED all first_oa_year / oa_date logic. Pilot (239 repository locations) showed oa_date == publication
    date (lag mean -0.01, SD 0.09, 0% opened >1y after publication), so it cannot date first availability.
  * Unpaywall now collects only: oa_status, has_repository_copy, oa_locations host_type/version.
  * [M5] NEW: paper-level deduplication (Lens ID). The file has 12,302 rows but ~10,114 unique Lens IDs, i.e. the
    unit is paper x institution. H1, H2a, H3 are re-estimated on one row per paper.
  * [O1] rewritten: Lens colour vs Unpaywall status agreement; repository-copy association (R2-8);
    Green split by repository version (R2-9); Gold with/without repository copy.
  * ZTNB estimator kept; alpha is at the upper bound (log-series limit) -> sensitivity only.
  * facts(): tolerant to raw vs. with_cluster file column names.

Blocks: M1 step3 | M2 step4 | M3 ZTNB | M4 data facts | M5 dedup robustness | O1 Unpaywall | O2 hurdle (needs full file)
NOT included: stacked DiD, Bartik IV, dose-response (pre_field_oa x Post has no OA term; does not test premium compression).

Run:
  RV_RAW=./outputs/Transport_CN_Scholarly_Works_with_cluster.csv RV_OUT=./outputs/final \
  RV_BOOT=999 RV_BOOT_ATT=300 python3 -u step5_full_pipeline.py
Optional: RV_FETCH=1 RV_EMAIL=you@univ.edu   (Unpaywall fetch, cached; resumable)
          RV_RAW_FULL=<csv incl. zero-patent papers with `cluster`>

KNOWN ISSUES (kept as-run so outputs reproduce; see docs/KNOWN_ISSUES.md):
  * uw_block(): the printed 'colour exact match' (47.4%) counts Lens 'closed' rows as mismatches.
    Use the cross-tabulation in results/unpaywall/lens_colour_vs_unpaywall_status.csv (88.1% / 83.3%).
  * uw_block(): 'g_oth' pools Green papers with accepted/published repository copies AND Green papers with no
    version information (n=299); its label 'accepted/published' is therefore inaccurate.
"""
import os, sys, re, json, hashlib, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("RV_OUT", "./outputs/final")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy.stats as ss
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm
import step3_primary_plan as s3
import step4_revised_H2H3 as s4

OUT = s3.OUT; YMIN, YMAX = s3.YMIN, s3.YMAX
EXTRA = []
ALPHAS = []
PLAN5 = {
    "M3": "ZTNB (NB2, cluster+year FE, analytic scores, cluster sandwich, t[G-1]); sensitivity only (alpha at bound)",
    "M4": "descriptive facts only; no hypothesis tests",
    "M5": "one row per Lens ID; re-estimate H1 (total, +lc), H2a interaction, H3 M0/M1 Green/Gold with ZTP",
    "O1": "Unpaywall: oa_status, has_repository_copy, repository versions; no date variables (oa_date not informative)",
    "O2": "hurdle logit(any patent citation) on full file incl. zero-citation papers; requires cluster column",
    "status": "All blocks are sensitivity/robustness; primary tests remain P1-P7 from Steps 3-4. Hypotheses redefined after review.",
}
PH5 = hashlib.sha256(json.dumps(PLAN5, sort_keys=True).encode()).hexdigest()


# ───────────────────────── ZTNB ─────────────────────────
def _terms(X, y, th):
    b, la = th[:-1], th[-1]; r = np.exp(-la)
    mu = np.exp(np.clip(X @ b, -20, 20)); lr = np.log(r) - np.log(r + mu); lp0 = r * lr
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * lr + y * (np.log(mu) - np.log(r + mu))
          - np.log(-np.expm1(lp0)))
    return ll, mu, r, lp0


def _scores(X, y, th):
    ll, mu, r, lp0 = _terms(X, y, th)
    s_eta = r * (y - mu) / (r + mu) - np.exp(lp0) * r * mu / ((r + mu) * (-np.expm1(lp0)))
    h = 1e-5; tp = th.copy(); tm = th.copy(); tp[-1] += h; tm[-1] -= h
    s_la = (_terms(X, y, tp)[0] - _terms(X, y, tm)[0]) / (2 * h)
    return np.column_stack([X * s_eta[:, None], s_la])


def ztnb_fit(d, xvars, fe=("cluster", "year")):
    d = d[d["y"] >= 1].dropna(subset=xvars).copy()
    M = s3.design(d, xvars, fe); names = list(M.columns); X = M.values; y = d["y"].values
    gi = pd.factorize(d["cluster"])[0]; G = gi.max() + 1; n, k = X.shape
    b0 = s3._newton(X, y, s3._start(X, y))
    f = lambda th: -_terms(X, y, th)[0].sum()
    g = lambda th: -_scores(X, y, th).sum(0)
    best = None
    for la0 in (-1.0, 1.0, 3.0, 5.0):
        o = minimize(f, np.r_[b0, la0], jac=g, method="L-BFGS-B", bounds=[(None, None)] * k + [(-7, 10)],
                     options=dict(maxiter=5000, maxfun=20000, ftol=1e-13, gtol=1e-8))
        if best is None or o.fun < best.fun:
            best = o
    th = best.x
    for _ in range(15):
        Jn = np.zeros((k + 1, k + 1))
        for i in range(k + 1):
            h = 1e-5 * max(1, abs(th[i])); tp = th.copy(); tm = th.copy(); tp[i] += h; tm[i] -= h
            Jn[:, i] = (_scores(X, y, tp).sum(0) - _scores(X, y, tm).sum(0)) / (2 * h)
        step = np.linalg.solve(-(Jn + Jn.T) / 2 + 1e-8 * np.eye(k + 1), _scores(X, y, th).sum(0))
        t_, f0 = 1.0, f(th)
        while t_ > 1e-4:
            cand = th + t_ * step; cand[-1] = min(max(cand[-1], -7), 10)
            if f(cand) <= f0: break
            t_ /= 2
        if np.max(np.abs(t_ * step)) < 1e-8 or f(cand) > f0: break
        th = cand
    gmax = np.max(np.abs(g(th))) / n
    ALPHAS.append(dict(xvars="+".join(xvars), n=n, alpha=float(np.exp(th[-1])), loglik=-f(th),
                       max_grad_per_n=gmax, la_at_bound=bool(th[-1] > 9.9)))
    if gmax > 1e-3 or th[-1] > 9.9:
        EXTRA.append(f"WARNING ZTNB: max|grad|/n={gmax:.2e}, alpha={np.exp(th[-1]):.3g} (n={n}, xvars={xvars}); "
                     "alpha at upper bound -> report ZTNB as direction check only")
    J = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        h = 1e-5 * max(1, abs(th[i])); tp = th.copy(); tm = th.copy(); tp[i] += h; tm[i] -= h
        J[:, i] = (_scores(X, y, tp).sum(0) - _scores(X, y, tm).sum(0)) / (2 * h)
    H = -(J + J.T) / 2; Hinv = np.linalg.pinv(H)
    S = _scores(X, y, th); Sg = np.zeros((G, k + 1)); np.add.at(Sg, gi, S)
    c = (G / (G - 1)) * ((n - 1) / (n - k - 1)); V = c * Hinv @ (Sg.T @ Sg) @ Hinv
    crit = ss.t.ppf(.975, G - 1); coefs = {}
    for t_ in xvars:
        if t_ in names:
            j = names.index(t_); se = np.sqrt(V[j, j]); b = th[j]
            coefs[t_] = dict(beta=b, se=se, lo=b - crit * se, hi=b + crit * se,
                             p=2 * ss.t.sf(abs(b / se), G - 1), p_wcb=np.nan)
    return dict(b=th[:-1], V=V[:-1, :-1], names=names, G=G, n=n, coefs=coefs, alpha=float(np.exp(th[-1])))


def ztnb_block(D):
    D = D.copy(); mu = D["lc"].mean(); D["lc_c"] = D["lc"] - mu; D["oa_x_lc"] = D["oa"] * D["lc_c"]
    r = ztnb_fit(D, ["oa", "lc_c", "oa_x_lc"])
    s3.rec("M3", f"ZTNB H2a OA x centred lc (alpha={r['alpha']:.2f})", "oa_x_lc", r["coefs"]["oa_x_lc"], r)
    for q in (.1, .5, .9):
        cq = D["lc"].quantile(q)
        s3.rec("M3", f"ZTNB OA premium at lc {q:.0%} pct", "oa_at_q", s3.lincom(r, {"oa": 1, "oa_x_lc": cq - mu}), r)
    Dp = D[D.year <= 2019].copy(); Dp["oa_x_yr"] = Dp["oa"] * (Dp["year"] - 2011)
    r = ztnb_fit(Dp, ["oa", "oa_x_yr"]); s3.rec("M3", "ZTNB H2b OA x year per 5y (pub<=2019)", "oa_x_yr", s4.scaled(r["coefs"]["oa_x_yr"], 5), r)
    d = D[D.colour != "other"]; cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"]; gaps = {}
    for tag, extra in [("M0", []), ("M1 + lc", ["lc"])]:
        r = ztnb_fit(d, cols + extra)
        for c in cols: s3.rec("M3", f"ZTNB {tag}", c, r["coefs"][c], r)
        gg = s3.lincom(r, {"c_green": 1, "c_gold": -1}); s3.rec("M3", f"ZTNB {tag} - Green/Gold", "green_minus_gold", gg, r)
        s3.rec("M3", f"ZTNB {tag} - Bronze/Gold", "bronze_minus_gold", s3.lincom(r, {"c_bronze": 1, "c_gold": -1}), r)
        gaps[tag] = gg["beta"]
    EXTRA.append(f"ZTNB Green/Gold log-gap: M0={gaps['M0']:.3f} -> M1={gaps['M1 + lc']:.3f} (no bootstrap CI for ZTNB)")


# ───────────────────────── M4: facts ─────────────────────────
def facts(raw):
    ycol = "Citing_Patents" if "Citing_Patents" in raw.columns else "Citing Patents Count"
    ycol_yr = "Pub_Year" if "Pub_Year" in raw.columns else "Publication Year"
    y = pd.to_numeric(raw[ycol], errors="coerce"); yr = pd.to_numeric(raw[ycol_yr], errors="coerce")
    L = [f"rows in file: {len(raw):,}; Citing_Patents==0: {(y == 0).sum():,}; missing: {y.isna().sum():,}"]
    b = {"<2003": (yr < 2003).sum(), "2003-2024": yr.between(2003, 2024).sum(), "2025": (yr == 2025).sum(),
         ">2025": (yr > 2025).sum(), "missing year": yr.isna().sum()}
    L.append("sample-window table (replaces manuscript Table A4): " + "; ".join(f"{k}={v:,}" for k, v in b.items()))
    t = pd.DataFrame({"year": yr, "eq1": (y == 1)}).dropna(subset=["year"]).groupby("year")["eq1"].agg(["mean", "size"])
    t.to_csv(OUT / "FACT_share_eq1_by_year.csv")
    L.append(f"share with exactly 1 citing patent: {(y[y >= 1] == 1).mean():.3f}")
    if "Lens ID" in raw.columns:
        L.append(f"unique Lens IDs: {raw['Lens ID'].nunique():,} of {len(raw):,} rows -> observation unit is paper x institution")
    if "Institution" in raw.columns:
        vc = raw["Institution"].value_counts(); vc.to_csv(OUT / "FACT_institutions_all.csv")
        L.append(f"distinct Institution values: {vc.shape[0]} (rows per institution in FACT_institutions_all.csv)")
    return L


# ───────────────────────── M5: dedup robustness ─────────────────────────
def dedup_block(D):
    if "Lens ID" not in D.columns:
        EXTRA.append("M5 skipped: no 'Lens ID' column"); return
    Dd = D.sort_values(["Lens ID", "Institution"] if "Institution" in D.columns else ["Lens ID"]).drop_duplicates("Lens ID", keep="first").copy()
    EXTRA.append(f"M5: rows {len(D):,} -> unique papers (Lens ID) {len(Dd):,}; OA share rows {D.oa.mean():.3f} vs papers {Dd.oa.mean():.3f}")
    g = D.groupby("Lens ID").agg(ny=("y", "nunique"), nc=("cluster", "nunique"), no=("oa", "nunique"))
    EXTRA.append(f"M5 sanity: Lens IDs with differing y/cluster/oa across rows: {(g.ny>1).sum()}/{(g.nc>1).sum()}/{(g.no>1).sum()}")
    r = s3.ztp_fit(Dd, ["oa"], boot=("oa",)); s3.rec("M5", "DEDUP ZTP H1 total (no lc)", "oa", r["coefs"]["oa"], r)
    r = s3.ztp_fit(Dd, ["oa", "lc"], boot=("oa",)); s3.rec("M5", "DEDUP ZTP H1 + lc", "oa", r["coefs"]["oa"], r)
    mu = Dd["lc"].mean(); Dd["lc_c"] = Dd["lc"] - mu; Dd["oa_x_lc"] = Dd["oa"] * Dd["lc_c"]
    r = s3.ztp_fit(Dd, ["oa", "lc_c", "oa_x_lc"], boot=("oa_x_lc",)); s3.rec("M5", "DEDUP ZTP H2a OA x centred lc", "oa_x_lc", r["coefs"]["oa_x_lc"], r)
    d3 = Dd[Dd.colour != "other"]; cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"]
    for tag, extra in [("M0", []), ("M1 + lc", ["lc"])]:
        r = s3.ztp_fit(d3, cols + extra)
        s3.rec("M5", f"DEDUP ZTP {tag} - Green/Gold", "green_minus_gold", s3.lincom(r, {"c_green": 1, "c_gold": -1}), r)
        s3.rec("M5", f"DEDUP ZTP {tag} - Bronze/Gold", "bronze_minus_gold", s3.lincom(r, {"c_bronze": 1, "c_gold": -1}), r)
    Dd.to_csv(OUT / "FACT_dedup_papers.csv", index=False)


# ───────────────────────── O1: Unpaywall (no dates) ─────────────────────────
def parse_uw(j):
    locs = j.get("oa_locations") or []
    rep = [l for l in locs if l.get("host_type") == "repository"]
    return dict(uw_is_oa=j.get("is_oa"), uw_status=j.get("oa_status"), uw_has_repo=j.get("has_repository_copy"),
                uw_versions=";".join(sorted({str(l.get("version")) for l in locs if l.get("version")})),
                uw_repo_versions=";".join(sorted({str(l.get("version")) for l in rep if l.get("version")})),
                uw_hosts=";".join(sorted({str(l.get("host_type")) for l in locs if l.get("host_type")})),
                uw_n_repo=len(rep))


def fetch_uw(dois, email, cache):
    if not email:
        raise SystemExit("RV_EMAIL must be set when RV_FETCH=1")
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["doi_l"])
    done = set(have["doi_l"]); todo = [x for x in dois if x not in done]
    print(f"Unpaywall: {len(todo):,} DOIs to fetch ({len(have):,} cached)")

    def one(doi):
        url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
        for k in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return dict(doi_l=doi, **parse_uw(json.load(r)))
            except urllib.error.HTTPError as e:
                if e.code == 404: return dict(doi_l=doi, uw_status="not_found")
                time.sleep(2 ** k)
            except Exception:
                time.sleep(2 ** k)
        return None
    rows = []
    with ThreadPoolExecutor(5) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            if r: rows.append(r)
            if (i + 1) % 500 == 0:
                pd.concat([have, pd.DataFrame(rows)]).to_csv(cache, index=False); print(f"  {i+1}/{len(todo)}")
    out = pd.concat([have, pd.DataFrame(rows)]).drop_duplicates("doi_l"); out.to_csv(cache, index=False); return out


def attach_uw(D_all, raw_cols):
    dc = next((c for c in raw_cols if c.strip().lower() == "doi"), None)
    if dc is None:
        EXTRA.append("O1 skipped: no DOI column"); return D_all
    D_all["doi_l"] = D_all[dc].astype(str).str.lower().str.strip().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    cache = OUT / "unpaywall_cache.csv"
    if os.environ.get("RV_FETCH") == "1":
        fetch_uw(sorted(set(D_all["doi_l"].dropna()) - {"nan", ""}), os.environ.get("RV_EMAIL", ""), cache)
    if cache.exists():
        uw = pd.read_csv(cache)
        uw = uw.drop(columns=[c for c in ("first_oa_year", "uw_oa_date", "uw_best_version", "uw_best_host") if c in uw.columns])
        D_all = D_all.merge(uw.drop_duplicates("doi_l"), on="doi_l", how="left")
        if "uw_has_repo" in D_all.columns:
            D_all["uw_repo"] = D_all["uw_has_repo"].astype(str).str.lower().eq("true").astype(int)
            D_all.loc[D_all["uw_status"].isna(), "uw_repo"] = np.nan
    return D_all


def uw_block(D):
    if "uw_status" not in D.columns:
        EXTRA.append("O1: no Unpaywall data (run with RV_FETCH=1 RV_EMAIL=...)"); return
    cov = D["uw_status"].notna().mean() if len(D) else 0
    EXTRA.append(f"O1 coverage: Unpaywall record for {cov:.1%} of rows ('not_found' counted as covered)")
    d = D[D.uw_status.notna() & (D.uw_status != "not_found")].copy()
    if len(d) < 500:
        EXTRA.append("O1: <500 usable Unpaywall records; analyses skipped"); return
    ct = pd.crosstab(d["colour"], d["uw_status"]); ct.to_csv(OUT / "FACT_lens_colour_vs_unpaywall_status.csv")
    uw_oa = (d["uw_status"] != "closed").astype(int)
    EXTRA.append(f"O1 agreement: binary OA (Lens vs Unpaywall) = {(uw_oa == d['oa']).mean():.1%}; "
                 f"colour exact match (gold/green/hybrid/bronze) = "
                 f"{(d['colour'].where(d['colour'].isin(['gold','green','hybrid','bronze']), 'x') == d['uw_status']).mean():.1%} (see FACT_lens_colour_vs_unpaywall_status.csv)")
    if "uw_repo" in d.columns:
        oa = d[d.oa == 1]
        EXTRA.append(f"O1: share of Lens-OA papers with a repository copy = {oa['uw_repo'].mean():.1%}; "
                     f"among Gold: {d[d.colour=='gold']['uw_repo'].mean():.1%}; among Green: {d[d.colour=='green']['uw_repo'].mean():.1%}")
        for lab, dd in [("all papers", d), ("Lens-OA papers only", oa)]:
            for tag, xv in [("no lc", ["uw_repo"]), ("+ lc", ["uw_repo", "lc"])]:
                r = s3.ztp_fit(dd, xv, boot=("uw_repo",)); s3.rec("O1", f"ZTP repository copy ({lab}, {tag})", "uw_repo", r["coefs"]["uw_repo"], r)
        gd = d[d.colour == "gold"]
        if gd["uw_repo"].nunique() == 2 and gd["uw_repo"].sum() >= 30:
            r = s3.ztp_fit(gd, ["uw_repo", "lc"]); s3.rec("O1", "ZTP Gold only: repository copy vs none (+ lc)", "uw_repo", r["coefs"]["uw_repo"], r)
    if "uw_repo_versions" in d.columns:
        g = d[d.colour == "green"]
        EXTRA.append(f"O1 Green repository versions: {g['uw_repo_versions'].fillna('none').value_counts().head(6).to_dict()}")
        rv = d["uw_repo_versions"].fillna("")
        sub_only = rv.str.contains("submittedVersion") & ~rv.str.contains("acceptedVersion|publishedVersion")
        dd = d[d.colour.isin(["closed", "gold", "green", "hybrid", "bronze"])].copy()
        dd["g_sub"] = ((dd.colour == "green") & sub_only.loc[dd.index]).astype(int)
        dd["g_oth"] = ((dd.colour == "green") & ~sub_only.loc[dd.index]).astype(int)
        if dd["g_sub"].sum() >= 30 and dd["g_oth"].sum() >= 30:
            cols = ["c_gold", "g_oth", "g_sub", "c_hybrid", "c_bronze"]
            r = s3.ztp_fit(dd, cols + ["lc"])
            for c in cols: s3.rec("O1", "ZTP M1 with Green split by repository version", c, r["coefs"][c], r)
            s3.rec("O1", "ZTP M1: Green(preprint-only)/Gold", "gsub_minus_gold", s3.lincom(r, {"g_sub": 1, "c_gold": -1}), r)
            s3.rec("O1", "ZTP M1: Green(accepted/published copy)/Gold", "goth_minus_gold", s3.lincom(r, {"g_oth": 1, "c_gold": -1}), r)
        else:
            EXTRA.append(f"O1: version split skipped (preprint-only Green n={int(dd['g_sub'].sum())}, other Green n={int(dd['g_oth'].sum())})")
    EXTRA.append("O1 caveat: Unpaywall status is as of query date; 'submittedVersion' labels can be inaccurate; no access-date information.")


# ───────────────────────── O2: hurdle ─────────────────────────
def hurdle(path):
    old = s3.RAW; s3.RAW = Path(path)
    try:
        Df = s3.load()
    except SystemExit as e:
        EXTRA.append(f"O2 skipped: {e} (full file needs a `cluster` column)"); s3.RAW = old; return
    s3.RAW = old
    Df = Df[Df.year.between(YMIN, YMAX)].copy(); Df["any"] = (Df["y"] >= 1).astype(float)
    EXTRA.append(f"O2: full file N={len(Df):,}; share with >=1 citing patent = {Df['any'].mean():.3f}")
    gi = pd.factorize(Df["cluster"])[0]; G = gi.max() + 1; crit = ss.t.ppf(.975, G - 1)
    for tag, xv in [("hurdle logit any patent ~ OA", ["oa"]), ("... + lc", ["oa", "lc"])]:
        M = s3.design(Df, xv, ("cluster", "year")); m = sm.Logit(Df["any"].values, M.values).fit(disp=0, cov_type="cluster", cov_kwds={"groups": gi})
        j = list(M.columns).index("oa"); b, se = m.params[j], m.bse[j]
        s3.rec("O2", tag + " (beta = log-odds)", "oa", dict(beta=b, se=se, lo=b - crit * se, hi=b + crit * se,
               p=2 * ss.t.sf(abs(b / se), G - 1), p_wcb=np.nan), dict(n=len(Df), G=G))


# ───────────────────────── report ─────────────────────────
def report(facts_l):
    fp = s3.fp; L = []; w = L.append
    w(f"# FINAL REPORT (plan hashes: step3 `{s3.PLAN_HASH[:12]}`, step4 `{s4.PH[:12]}`, step5 `{PH5[:12]}`; sample {YMIN}-{YMAX})\n")
    keys = [k for k in ("P1_H1_total", "P2_H1_adj", "P3_H2_contrast", "P4_H3_gap", "P5_H2a_visibility", "P6_H2b_trend", "P7_H3_attenuation") if k in s3.PRIM]
    pv = lambda r: r["p_wcb"] if not pd.isna(r["p_wcb"]) else r["p"]
    glob = s3.holm(np.array([pv(s3.PRIM[k]) for k in keys]))
    f3 = [k for k in keys if k.startswith(("P1", "P2", "P3", "P4"))]; f4 = [k for k in keys if k.startswith(("P5", "P6", "P7"))]
    h3 = dict(zip(f3, s3.holm(np.array([s3.PRIM[k]["p"] for k in f3])))); h4 = dict(zip(f4, s3.holm(np.array([pv(s3.PRIM[k]) for k in f4]))))
    w("## Primary tests (Holm within family; last column Holm over all seven)\n")
    w("| test | spec | RR | 95% CI | p (t) | p (boot) | Holm (family) | Holm (all 7) |\n|---|---|---|---|---|---|---|---|")
    for k, g in zip(keys, glob):
        r = s3.PRIM[k]; hf = h3.get(k, h4.get(k))
        w(f"| {k} | {r['spec']} | {np.exp(r['beta']):.3f} | [{np.exp(r['lo']):.3f}, {np.exp(r['hi']):.3f}] | {fp(r['p'])} | {fp(r['p_wcb'])} | {fp(hf)} | {fp(g)} |")
    w("\nRR = rate ratio on the latent (untruncated) rate; not comparable to the manuscript's 9.4%. Bootstrap p floors at 1/(B+1).\n")
    t = pd.DataFrame(s3.ROWS)
    w("## Original vs deduplicated (one row per Lens ID)\n")
    w("| spec | rows RR [CI] | dedup RR [CI] |\n|---|---|---|")
    for nm, a, b in [("H1 total", "ZTP H1 total (no lc)", "DEDUP ZTP H1 total (no lc)"), ("H1 + lc", "ZTP H1 + lc", "DEDUP ZTP H1 + lc"),
                     ("H2a interaction", "ZTP OA x centred lc (RR per 1 log-unit of academic citations)", "DEDUP ZTP H2a OA x centred lc"),
                     ("H3 Green/Gold M0", "ZTP M0 colour only - Green minus Gold", "DEDUP ZTP M0 - Green/Gold"),
                     ("H3 Green/Gold M1", "ZTP M1 + lc - Green minus Gold", "DEDUP ZTP M1 + lc - Green/Gold")]:
        ra, rb = t[t.spec == a], t[t.spec == b]
        if len(ra) and len(rb):
            ra, rb = ra.iloc[0], rb.iloc[0]
            w(f"| {nm} | {ra.RR:.3f} [{ra.RR_lo:.3f}, {ra.RR_hi:.3f}] (p={fp(ra.p)}) | {rb.RR:.3f} [{rb.RR_lo:.3f}, {rb.RR_hi:.3f}] (p={fp(rb.p)}) |")
    w("\n## ZTP vs ZTNB (direction check only; alpha at bound)\n")
    w("| spec | ZTP RR [CI] | ZTNB RR [CI] |\n|---|---|---|")
    for nm, a, b in [("H1 total", "ZTP H1 total (no lc)", "H1 ZTNB total"), ("H1 + lc", "ZTP H1 + lc", "H1 ZTNB + lc"),
                     ("H2a interaction", "ZTP OA x centred lc (RR per 1 log-unit of academic citations)", "ZTNB H2a OA x centred lc"),
                     ("H3 Green/Gold (M1)", "ZTP M1 + lc - Green minus Gold", "ZTNB M1 + lc - Green/Gold"),
                     ("H3 Green/Gold (M0)", "ZTP M0 colour only - Green minus Gold", "ZTNB M0 - Green/Gold")]:
        ra = t[t.spec == a]; rb = t[t.spec.str.startswith(b)]
        if len(ra) and len(rb):
            ra, rb = ra.iloc[0], rb.iloc[0]
            w(f"| {nm} | {ra.RR:.3f} [{ra.RR_lo:.3f}, {ra.RR_hi:.3f}] | {rb.RR:.3f} [{rb.RR_lo:.3f}, {rb.RR_hi:.3f}] |")
    w("\n## Notes, warnings, optional-block status\n"); w("\n".join("- " + x for x in EXTRA + s4.EXTRA) + "\n")
    w("## Data facts for the response letter (M4)\n"); w("\n".join("- " + x for x in facts_l) + "\n")
    if ALPHAS:
        w("## ZTNB dispersion\n"); w(pd.DataFrame(ALPHAS).round(4).to_markdown(index=False) + "\n")
    w(f"## Multiplicity disclosure\nThis run fitted {len(t)} coefficient rows (plus Steps 1-4). State in Limitations that many specifications were "
      "examined, H2/H3 were redefined after review, H2b is post hoc, and the observation unit is paper x institution.\n")
    show = t[["block", "spec", "term", "n", "RR", "RR_lo", "RR_hi", "p", "p_wcb"]].copy()
    for c in ("RR", "RR_lo", "RR_hi"): show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["p"] = show["p"].map(fp); show["p_wcb"] = show["p_wcb"].map(fp)
    for blk in show["block"].unique():
        w(f"### Ledger: {blk}\n"); w(show[show.block == blk].drop(columns="block").to_markdown(index=False)); w("")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    s3.write_plan(); s4.write_plan()
    (OUT / "PLAN_STEP5.md").write_text(f"# Step 5 plan ({datetime.now():%Y-%m-%d %H:%M:%S})\nSHA-256 `{PH5}`\n\n" +
                                       "\n".join(f"- **{k}**: {v}" for k, v in PLAN5.items()), encoding="utf-8")
    s3.ztnb_fit = ztnb_fit
    raw = s3.read_csv(s3.RAW); raw.columns = raw.columns.str.strip()
    facts_l = facts(raw)
    D_all = attach_uw(s3.add_lag_share(s3.load()), list(raw.columns))
    D = D_all[D_all.year.between(YMIN, YMAX) & (D_all["y"] >= 1)].copy()
    print(f"N={len(D):,} clusters={D.cluster.nunique()} | plans hashed")
    print("[M1] step3 ..."); s3.run(D, D_all)
    print("[M2] step4 ..."); s4.h2a(D); s4.h2b(D); s4.h3(D)
    print("[M3] ZTNB ..."); ztnb_block(D)
    print("[M5] dedup ..."); dedup_block(D)
    print("[O1] Unpaywall ..."); uw_block(D)
    if os.environ.get("RV_RAW_FULL"):
        print("[O2] hurdle ..."); hurdle(os.environ["RV_RAW_FULL"])
    pd.DataFrame(s3.ROWS).to_csv(OUT / "ALL_LEDGER.csv", index=False)
    report(facts_l); print("done ->", OUT)


if __name__ == "__main__":
    main()
