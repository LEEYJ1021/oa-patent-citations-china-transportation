#!/usr/bin/env python3
"""
STEP 4 - REVISED H2 / H3 (heterogeneity by academic visibility; route gap explained by visibility)
Requires step3_primary_plan.py in the same folder (imports its zero-truncated engine).
Plan is written with a SHA-256 hash BEFORE estimation. All coefficients are reported.

  H2a (primary): OA premium increases with academic visibility  -> ZTP, OA x centred log(1+academic cit.)
  H2b (primary): OA premium trend across publication year, pub<=2019 (>=5y exposure) -> ZTP, OA x (year-2011)
                 + omnibus Wald test that the four period premiums are equal (all years)
  H3  (primary): the Green-Gold raw gap is reduced by adjusting for academic citations
                 -> cluster bootstrap (resample clusters) of gap(M0) - gap(M1), log-rate-ratio scale
  Secondary: OA premium at lc 10/50/90th pct, by visibility tercile, route premiums, Bronze vs Gold.
Env: RV_RAW, RV_OUT (./outputs/step4), RV_BOOT (score bootstrap, 999), RV_BOOT_ATT (cluster bootstrap, 300)
"""
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime
os.environ.setdefault("RV_OUT", "./outputs/step4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy.stats as ss
import step3_primary_plan as s3

OUT = s3.OUT; YMIN, YMAX = s3.YMIN, s3.YMAX
B_ATT = int(os.environ.get("RV_BOOT_ATT", 300)); RNG = np.random.default_rng(4242)

PLAN4 = {
    "status": "Hypotheses H2/H3 redefined after peer review and after Step 3 results; H2b is a post-hoc single-df alternative to the underpowered endpoint contrast (endpoint contrast still reported).",
    "H2a": "ZTP (cluster+year FE): patents ~ OA + lc_c + OA x lc_c ; lc_c = log(1+academic citations) centred. Primary parameter: OA x lc_c (rate-ratio per 1 log-unit).",
    "H2b": "ZTP on pub<=2019: patents ~ OA + OA x (year-2011); primary parameter reported per 5 years. Omnibus Wald F test of equal period premiums (4 periods, all years, no lc).",
    "H3": "M0: colour dummies; M1: + lc (closed reference, 'other' dropped). Statistic: log(Green/Gold)_M0 - log(Green/Gold)_M1 with cluster-bootstrap percentile CI; share explained = 1 - gap_M1/gap_M0.",
    "multiplicity": "Holm over H2a, H2b, H3 (bootstrap/cluster p).",
    "interpretation_limits": "lc is cumulative and may be post-OA: results are heterogeneity/adjusted associations, not mechanisms. OA status and cluster shares are retrospective.",
}
PH = hashlib.sha256(json.dumps(PLAN4, sort_keys=True).encode()).hexdigest()
EXTRA = []


def write_plan():
    L = [f"# Step 4 plan (written {datetime.now():%Y-%m-%d %H:%M:%S}, before estimation)", f"SHA-256: `{PH}`\n"]
    L += [f"- **{k}**: {v}" for k, v in PLAN4.items()]
    (OUT / "STEP4_PLAN.md").write_text("\n".join(L), encoding="utf-8")


def scaled(r, f):
    return dict(beta=r["beta"] * f, se=r["se"] * f, lo=r["lo"] * f, hi=r["hi"] * f, p=r["p"], p_wcb=r["p_wcb"])


def h2a(D):
    D = D.copy(); mu = D["lc"].mean(); D["lc_c"] = D["lc"] - mu; D["oa_x_lc"] = D["oa"] * D["lc_c"]
    res = s3.ztp_fit(D, ["oa", "lc_c", "oa_x_lc"], boot=("oa_x_lc",))
    s3.rec("H2a", "ZTP OA x centred lc (RR per 1 log-unit of academic citations)", "oa_x_lc", res["coefs"]["oa_x_lc"], res, "P5_H2a_visibility")
    s3.rec("H2a", "ZTP OA premium at mean visibility", "oa", res["coefs"]["oa"], res)
    for q in (.1, .5, .9):
        cq = D["lc"].quantile(q)
        s3.rec("H2a", f"ZTP OA premium at lc {q:.0%} percentile (lc={cq:.2f})", "oa_at_q",
               s3.lincom(res, {"oa": 1, "oa_x_lc": cq - mu}), res)
    D["lc_t"] = pd.qcut(D["lc"], 3, labels=False, duplicates="drop")
    for t in sorted(D["lc_t"].unique()):
        r2 = s3.ztp_fit(D[D.lc_t == t], ["oa"], boot=("oa",))
        s3.rec("H2a", f"ZTP OA premium, visibility tercile {int(t)+1} (1=lowest)", "oa", r2["coefs"]["oa"], r2)


def h2b(D):
    D = D.copy()
    for i, (a, b) in enumerate(s3.PERIODS):
        D[f"oa_P{i}"] = D["oa"] * D["year"].between(a, b).astype(int)
    xv = [f"oa_P{i}" for i in range(4)]
    res = s3.ztp_fit(D, xv)
    idx = [res["names"].index(k) for k in xv]; b = res["b"][idx]; V = res["V"][np.ix_(idx, idx)]
    R = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]], float)
    Wd = (R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b); F = Wd / 3
    EXTRA.append(f"Omnibus Wald test of equal period premiums (2003-09/2010-14/2015-19/2020-24, ZTP, no lc): F(3,{res['G']-1})={F:.2f}, p={ss.f.sf(F, 3, res['G']-1):.3f}")
    Dp = D[D.year <= 2019].copy(); Dp["oa_x_yr"] = Dp["oa"] * (Dp["year"] - 2011)
    for tag, extra, prim in [("no lc", [], "P6_H2b_trend"), ("+ lc", ["lc"], None)]:
        r = s3.ztp_fit(Dp, ["oa", "oa_x_yr"] + extra, boot=("oa_x_yr",))
        s3.rec("H2b", f"ZTP pub<=2019 OA x year, per 5 years ({tag})", "oa_x_yr", scaled(r["coefs"]["oa_x_yr"], 5), r, prim)


def beta_map(d, xv):
    M = s3.design(d, xv, ("cluster", "year")); X = M.values; y = d["y"].values
    return dict(zip(M.columns, s3._newton(X, y, s3._start(X, y))))


def gaps(d, cols):
    b0 = beta_map(d, cols); b1 = beta_map(d, cols + ["lc"])
    g0 = b0["c_green"] - b0["c_gold"]; g1 = b1["c_green"] - b1["c_gold"]
    return g0, g1, g0 - g1, 1 - g1 / g0


def h3(D):
    d = D[D.colour != "other"].copy(); cols = ["c_gold", "c_green", "c_hybrid", "c_bronze"]
    for tag, extra in [("M0 colour only", []), ("M1 + lc", ["lc"])]:
        res = s3.ztp_fit(d, cols + extra, boot=("c_green", "c_gold"))
        for c in cols:
            s3.rec("H3", f"ZTP {tag}", c, res["coefs"][c], res)
        s3.rec("H3", f"ZTP {tag} - Green/Gold", "green_minus_gold", s3.lincom(res, {"c_green": 1, "c_gold": -1}), res)
        s3.rec("H3", f"ZTP {tag} - Bronze/Gold", "bronze_minus_gold", s3.lincom(res, {"c_bronze": 1, "c_gold": -1}), res)
    g0, g1, dif, share = gaps(d, cols); cl = d["cluster"].unique(); bs = []
    for _ in range(B_ATT):
        pick = RNG.choice(cl, size=len(cl), replace=True); parts = []
        for i, c in enumerate(pick):
            g = d[d.cluster == c].copy(); g["cluster"] = f"b{i}"; parts.append(g)
        try:
            bs.append(gaps(pd.concat(parts), cols))
        except Exception:
            continue
    bs = np.array(bs); dd = bs[:, 2]
    p = 2 * min((np.sum(dd <= 0) + 1) / (len(dd) + 1), (np.sum(dd >= 0) + 1) / (len(dd) + 1))
    r = dict(beta=dif, se=dd.std(), lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5), p=min(p, 1), p_wcb=np.nan)
    s3.rec("H3", "cluster bootstrap: log gap(M0) - log gap(M1) (RR col = gap(M0)/gap(M1) ratio)", "gap_reduction", r,
           dict(n=len(d), G=len(cl)), "P7_H3_attenuation")
    sh = bs[:, 3]
    EXTRA.append(f"H3: Green/Gold log-gap M0={g0:.3f} (RR {np.exp(g0):.2f}), M1={g1:.3f} (RR {np.exp(g1):.2f}); "
                 f"share of raw gap explained by academic citations = {share*100:.0f}% (bootstrap 95% CI [{np.nanpercentile(sh,2.5)*100:.0f}%, {np.nanpercentile(sh,97.5)*100:.0f}%], "
                 f"B={len(bs)}; unstable if gap(M0) is near 0 in draws)")


def report():
    fp = s3.fp; L = []; w = L.append
    w(f"# STEP 4 report (plan hash `{PH[:16]}...`; sample {YMIN}-{YMAX})\n")
    keys = [k for k in ("P5_H2a_visibility", "P6_H2b_trend", "P7_H3_attenuation") if k in s3.PRIM]
    ph = s3.holm(np.array([s3.PRIM[k]["p_wcb"] if not pd.isna(s3.PRIM[k]["p_wcb"]) else s3.PRIM[k]["p"] for k in keys]))
    w("## Primary family (Holm over three tests; p = wild-score/bootstrap p where available)\n")
    w("| test | spec | RR | 95% CI | p (t) | p (boot) | Holm p |\n|---|---|---|---|---|---|---|")
    for k, a in zip(keys, ph):
        r = s3.PRIM[k]
        w(f"| {k} | {r['spec']} | {np.exp(r['beta']):.3f} | [{np.exp(r['lo']):.3f}, {np.exp(r['hi']):.3f}] | {fp(r['p'])} | {fp(r['p_wcb'])} | {fp(a)} |")
    w("\n" + "\n".join("- " + x for x in EXTRA) + "\n")
    t = pd.DataFrame(s3.ROWS)[["block", "spec", "term", "n", "RR", "RR_lo", "RR_hi", "p", "p_wcb"]].copy()
    for c in ("RR", "RR_lo", "RR_hi"): t[c] = t[c].map(lambda x: f"{x:.3f}")
    t["p"] = t["p"].map(fp); t["p_wcb"] = t["p_wcb"].map(fp)
    for blk in ("H2a", "H2b", "H3"):
        w(f"## {blk} - all fitted coefficients\n"); w(t[t.block == blk].drop(columns="block").to_markdown(index=False)); w("")
    w("## Reading rules\n- Hypotheses redefined after review/Step 3; H2b post hoc. Report the Step 3 endpoint contrast alongside.\n"
      "- Academic citations are cumulative and possibly post-OA: heterogeneity/adjusted association, not mechanism.\n"
      "- Non-significant = 'not distinguishable from zero' with CI.")
    (OUT / "STEP4_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    write_plan()
    D_all = s3.add_lag_share(s3.load())
    D = D_all[D_all.year.between(YMIN, YMAX) & (D_all["y"] >= 1)].copy()
    print(f"plan {PH[:12]} | N={len(D):,}")
    h2a(D); h2b(D); h3(D)
    pd.DataFrame(s3.ROWS).to_csv(OUT / "S4_ALL_COEFFICIENTS.csv", index=False)
    report(); print("done ->", OUT)


if __name__ == "__main__":
    main()
