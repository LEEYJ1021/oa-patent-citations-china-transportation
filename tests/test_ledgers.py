"""Consistency checks on the curated result tables (no licensed data required)."""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "results"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def test_sample_flow_arithmetic():
    f = pd.read_csv(R / "primary/sample_flow.csv").set_index("step")["records"]
    assert f["raw export rows"] + f["empty padding rows"] + f["missing patent-citation count"] == f["= records after cleaning"] == 12348
    assert f["= records after cleaning"] + f["missing publication year"] + f["published 2026"] == f["= records with valid year"] == 12302
    assert f["= records with valid year"] + f["published before 2003"] + f["published 2025 (partial year)"] == f["= main analysis sample (2003-2024)"] == 12020
    assert f["= main analysis sample (2003-2024)"] + f["records with unclassified colour ('other')"] == f["= colour-model sample"] == 11955


def test_primary_tests_holm_all7():
    t = pd.read_csv(R / "primary/primary_tests_P1_P7.csv")
    p = np.where(t.p_boot.notna(), t.p_boot, pd.to_numeric(t.p_t.replace("<.001", "0.0005"), errors="coerce"))
    adj = holm(p)
    for a, b in zip(adj, t.holm_all7):
        assert abs(a - float(b)) < 0.006, (a, b)


def test_rate_ratio_ordering():
    for _, r in pd.concat([pd.read_csv(R / "primary/h1_ledger.csv"), pd.read_csv(R / "primary/h3_colour_ledger.csv")]).iterrows():
        assert r.RR_lo <= r.RR <= r.RR_hi


def test_unpaywall_agreement():
    ct = pd.read_csv(R / "unpaywall/lens_colour_vs_unpaywall_status.csv", index_col=0)
    cols = ["bronze", "closed", "gold", "green", "hybrid"]
    diag = sum(ct.loc[c, c] for c in cols)
    assert ct.values.sum() == 11905 and diag == 10488
    oa = ["bronze", "gold", "green", "hybrid"]
    assert abs(sum(ct.loc[c, c] for c in oa) / ct.loc[oa].values.sum() - 0.833) < 0.001


def test_leave_one_institution_out_summary():
    s = pd.read_csv(R / "robustness/C3_leave_one_institution_out_summary.csv").set_index("spec")
    assert (s.n_total == 33).all()
    assert s.loc["H1 OA + lc", "n_sig"] == 33 and s.loc["H2a OA x lc_c", "n_sig"] == 26 and s.loc["H3 Green/Gold M1", "n_sig"] == 1


def test_dedup_conclusions_unchanged():
    d = pd.read_csv(R / "robustness/rows_vs_dedup.csv")
    assert ((d.papers_lo > 1) == (d.rows_lo > 1)).sum() >= 4   # significance pattern preserved for 4 of 5 (H3 M1 stays n.s.)
