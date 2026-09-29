#!/usr/bin/env python3
"""
STEP 6 - REMAINING ANALYSES + INSTITUTION HARMONISATION
=======================================================
Needs in the same folder: step3_primary_plan.py, step4_revised_H2H3.py, step5_full_pipeline.py (patched: alpha bug fixed)

Blocks
  [A] ZTNB re-run with the corrected estimator (multi-start, alpha up to e^10, Newton polish, alpha reported correctly)
  [B] Institution harmonisation (Beihang = Beijing University of Aeronautics and Astronautics, etc.) + near-duplicate flags
  [C] Institution-level robustness for H1 / H2a / H3:
        C1 SE clustered by institution (instead of research cluster)
        C2 institution fixed effects added (cluster+year+institution), SE clustered by research cluster
        C3 leave-one-institution-out (H1, H2a, Green/Gold M1)
  [D] Unpaywall (RV_FETCH=1 RV_EMAIL=...; else uses unpaywall_cache.csv if present)   [step5 code]
  [E] Hurdle on the full file (RV_RAW_FULL=...; needs `cluster` column)                [step5 code]
Institution is NOT a variable in the primary models; blocks C are robustness only. Papers with several institutions are
assigned to the FIRST listed institution (primary institution) for FE/clustering/LOO - state this in the paper.

Run: RV_RAW=./outputs/Transport_CN_Scholarly_Works_with_cluster.csv RV_OUT=./outputs/step6 python3 -u step6_remaining.py
Env: RV_INST_COL (institution column; auto-detected otherwise), RV_LOO_MIN (min papers per institution for LOO, default 30)
"""
import os, sys, re, json, hashlib, difflib
from pathlib import Path
from datetime import datetime
os.environ.setdefault("RV_OUT", "./outputs/step6")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy.stats as ss
import step3_primary_plan as s3
import step4_revised_H2H3 as s4
import step5_full_pipeline as s5

OUT = s3.OUT; YMIN, YMAX = s3.YMIN, s3.YMAX; EXTRA = []
PLAN6 = {"A": "ZTNB corrected estimator for H1, H2a, H2b, H3 (M0/M1)",
         "B": "institution harmonisation by alias rules + normalisation; near-duplicate pairs flagged (difflib ratio>=0.88) for manual decision",
         "C": "robustness: SE clustered by institution; institution FE; leave-one-institution-out (n>=RV_LOO_MIN)",
         "D_E": "Unpaywall (open-at-publication, Green version) and hurdle, as in step5",
         "status": "robustness/sensitivity only; primary tests remain P1-P7"}
PH6 = hashlib.sha256(json.dumps(PLAN6, sort_keys=True).encode()).hexdigest()

ALIASES = [(r"beihang|beijing university of aeronautics|beijing univ\.? of aeronautics|buaa|北京航空航天", "Beihang University")]


def canon(name):
    low = str(name).strip().lower()
    for pat, tgt in ALIASES:
        if re.search(pat, low):
            return tgt
    low = re.sub(r"[^\w\s]", " ", low.replace("&", " and ")); low = re.sub(r"\s+", " ", low).strip()
    return low.title()


def harmonise(raw):
    col = os.environ.get("RV_INST_COL") or next((c for c in raw.columns if re.search(r"institut|affil|organi", c, re.I)), None)
    if col is None:
        raise SystemExit("no institution column; set RV_INST_COL")
    first = raw[col].astype(str).str.split(";").str[0].str.strip()
    can = first.map(canon)
    m = pd.DataFrame({"raw": first, "canonical": can}).groupby("canonical")["raw"].agg(lambda s: sorted(set(s)))
    merged = m[m.map(len) > 1]
    names = sorted(can.unique()); near = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= .88:
                near.append((a, b))
    tab = can.value_counts().rename_axis("institution").reset_index(name="papers"); tab.to_csv(OUT / "FACT_institutions_canonical.csv", index=False)
    EXTRA.append(f"[B] institution column '{col}': {first.nunique()} raw strings -> {can.nunique()} canonical institutions (primary institution = first listed)")
    EXTRA.append("[B] alias merges applied: " + (json.dumps(merged.to_dict(), ensure_ascii=False) if len(merged) else "none needed (Beihang already appears under one name)"))
    EXTRA.append("[B] near-duplicate pairs to decide by hand: " + (str(near) if near else "none") +
                 " | note: CAS and UCAS are kept as separate institutions (as in Table A1); merge them only if you decide so")
    EXTRA.append("[B] Table A1 must list each institution once; compare FACT_institutions_canonical.csv with the 33 rows of the manuscript table")
    return can


# ───────────────────────── C: institution robustness ─────────────────────────
def ztp_fit_g(d, xvars, fe, gcol):
    d = d[d["y"] >= 1].dropna(subset=xvars + [gcol]).copy()
    M = s3.design(d, xvars, fe); names = list(M.columns); X = M.values; y = d["y"].values
    gi = pd.factorize(d[gcol])[0]; G = gi.max() + 1; n, k = X.shape
    b = s3._newton(X, y, s3._start(X, y)); s, W, _ = s3._sw(X, y, b)
    Hinv = np.linalg.pinv(X.T @ (W[:, None] * X)); Sg = np.zeros((G, k)); np.add.at(Sg, gi, X * s[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * Hinv @ (Sg.T @ Sg) @ Hinv; crit = ss.t.ppf(.975, G - 1); coefs = {}
    for t_ in xvars:
        if t_ in names:
            j = names.index(t_); se = np.sqrt(V[j, j])
            coefs[t_] = dict(beta=b[j], se=se, lo=b[j] - crit * se, hi=b[j] + crit * se, p=2 * ss.t.sf(abs(b[j] / se), G - 1), p_wcb=np.nan)
    return dict(b=b, V=V, names=names, G=G, n=n, coefs=coefs)


def specs(D):
    D = D.copy(); mu = D["lc"].mean(); D["lc_c"] = D["lc"] - mu; D["oa_x_lc"] = D["oa"] * D["lc_c"]
    d3 = D[D.colour != "other"]
    return [("H1 OA total", D, ["oa"], "oa", None), ("H1 OA + lc", D, ["oa", "lc"], "oa", None),
            ("H2a OA x lc_c", D, ["oa", "lc_c", "oa_x_lc"], "oa_x_lc", None),
            ("H3 Green/Gold M1", d3, ["c_gold", "c_green", "c_hybrid", "c_bronze", "lc"], "gap", {"c_green": 1, "c_gold": -1})]


def inst_block(D, LOO_MIN):
    for nm, d, xv, term, lc in specs(D):
        variants = [("SE by institution", ("cluster", "year"), "inst"), ("institution FE, SE by research cluster", ("cluster", "year", "inst"), "cluster")]
        for vn, fe, g in variants:
            r = ztp_fit_g(d, xv, fe, g)
            rr = s3.lincom(r, lc) if lc else r["coefs"][term]
            s3.rec("C", f"{nm} | {vn} (G={r['G']})", "gap" if lc else term, rr, r)
    rows = []
    counts = D["inst"].value_counts(); keep = counts[counts >= LOO_MIN].index
    for inst in keep:
        d = D[D.inst != inst]
        for nm, dd, xv, term, lc in specs(d):
            if nm == "H1 OA total": continue
            r = s3.ztp_fit(dd, xv); rr = s3.lincom(r, lc) if lc else r["coefs"][term]
            rows.append(dict(dropped=inst, dropped_n=int(counts[inst]), spec=nm, RR=np.exp(rr["beta"]), lo=np.exp(rr["lo"]), hi=np.exp(rr["hi"]), p=rr["p"]))
    t = pd.DataFrame(rows); t.to_csv(OUT / "C3_leave_one_institution_out.csv", index=False)
    for nm, g in t.groupby("spec"):
        a, b = g.loc[g.RR.idxmin()], g.loc[g.RR.idxmax()]
        EXTRA.append(f"[C3] {nm}: LOO RR range {a.RR:.3f} (drop {a.dropped}) to {b.RR:.3f} (drop {b.dropped}); "
                     f"significant (p<.05) in {(g.p < .05).sum()}/{len(g)} leave-outs")
    return t


# ───────────────────────── report ─────────────────────────
def report():
    fp = s3.fp; L = []; w = L.append; t = pd.DataFrame(s3.ROWS)
    w(f"# STEP 6 REPORT (plan `{PH6[:12]}`; sample {YMIN}-{YMAX})\n")
    w("## Notes\n"); w("\n".join("- " + x for x in EXTRA + s5.EXTRA) + "\n")
    w("## ZTNB estimates (corrected): dispersion and convergence\n")
    a = pd.DataFrame(s5.ALPHAS); w(a.round(4).to_markdown(index=False) if len(a) else "none"); w("")
    w("alpha = NB2 dispersion (variance = mu + alpha*mu^2). Large alpha (tens) is expected here: 68% of papers have exactly one citing patent and the tail reaches 159. "
      "If la_at_bound is True the estimate sits on the upper bound (log-series limit); report the ZTNB as a sensitivity check only.\n")
    show = t[["block", "spec", "term", "n", "RR", "RR_lo", "RR_hi", "p", "p_wcb"]].copy()
    for c in ("RR", "RR_lo", "RR_hi"): show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["p"] = show["p"].map(fp); show["p_wcb"] = show["p_wcb"].map(fp)
    for blk in show["block"].unique():
        w(f"### Ledger: {blk}\n"); w(show[show.block == blk].drop(columns="block").to_markdown(index=False)); w("")
    w(f"Multiplicity: this run fitted {len(t)} coefficient rows; robustness only.")
    (OUT / "STEP6_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    (OUT / "PLAN_STEP6.md").write_text(f"# Step 6 plan ({datetime.now():%Y-%m-%d %H:%M:%S})\nSHA-256 `{PH6}`\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in PLAN6.items()), encoding="utf-8")
    raw = s3.read_csv(s3.RAW); raw.columns = raw.columns.str.strip()
    can = harmonise(raw)
    D_all = s3.add_lag_share(s3.load())
    # load() drops rows with missing keys; align institution by the same row filter
    keep = raw.dropna(subset=["Pub_Year", "Citing_Patents", "cluster"]).index
    D_all["inst"] = can.loc[keep].values
    D_all = s5.attach_uw(D_all, list(raw.columns))
    D = D_all[D_all.year.between(YMIN, YMAX) & (D_all["y"] >= 1)].copy()
    print(f"N={len(D):,} institutions={D.inst.nunique()}")
    s3.ztnb_fit = s5.ztnb_fit
    print("[A] ZTNB ..."); s5.ztnb_block(D)
    print("[C] institution robustness ..."); inst_block(D, int(os.environ.get("RV_LOO_MIN", 30)))
    print("[D] Unpaywall ..."); s5.uw_block(D)
    if os.environ.get("RV_RAW_FULL"):
        print("[E] hurdle ..."); s5.hurdle(os.environ["RV_RAW_FULL"])
    pd.DataFrame(s3.ROWS).to_csv(OUT / "STEP6_LEDGER.csv", index=False); report(); print("done ->", OUT)


if __name__ == "__main__":
    main()
