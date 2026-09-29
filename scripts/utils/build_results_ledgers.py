#!/usr/bin/env python3
"""
Rebuild the curated result ledgers in results/ from the numbers reported in the pipeline reports
(FINAL_REPORT.md of step5, STEP3/4/6 reports, and the step1/step2 exploratory reports).

Why this exists: the pipeline writes full ledgers (ALL_LEDGER.csv, S*_ALL_COEFFICIENTS.csv) when it is run on the
licensed Lens.org export. The files in results/ are the *transcribed, manuscript-facing* tables, so a reader can
check every number in the paper without the raw data. If you re-run the pipeline, compare its ledgers against these.

Usage (from the repository root):  python3 scripts/utils/build_results_ledgers.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("results"); nan = np.nan
cols = ["source", "block", "spec", "term", "n", "RR", "RR_lo", "RR_hi", "p", "p_wcb"]
L = []


def add(src, blk, rows):
    for r in rows:
        L.append((src, blk) + tuple(r))


add("step3", "H1", [
    ("ZTP H1 total (no lc)", "oa", 12020, 1.524, 1.361, 1.706, "<.001", 0.001),
    ("ZTP H1 + lc", "oa", 12020, 1.338, 1.211, 1.480, "<.001", 0.001),
    ("ZTP H1 + lc + pub-type FE + field tags", "oa", 12020, 1.374, 1.249, 1.511, "<.001", 0.001),
    ("ZTNB H1 total (superseded statsmodels estimator)", "oa", 12020, 1.666, 1.407, 1.974, "<.001", nan),
    ("ZTNB H1 + lc (superseded statsmodels estimator)", "oa", 12020, 1.414, 1.203, 1.662, "<.001", nan),
    ("ZTP H1 Gold/Hybrid vs closed (publisher-hosted), no lc", "oa_pub", 10553, 1.289, 1.074, 1.547, "0.008", 0.015),
    ("ZTP H1 Gold/Hybrid vs closed, + lc", "oa_pub", 10553, 1.258, 1.053, 1.502, "0.014", 0.029)])
add("step3", "H2", [
    ("ZTP period premium 2003-2009 (no lc)", "oa_P0", 12020, 2.109, 1.192, 3.732, "0.013", nan),
    ("ZTP period premium 2010-2014 (no lc)", "oa_P1", 12020, 1.601, 1.091, 2.352, "0.018", nan),
    ("ZTP period premium 2015-2019 (no lc)", "oa_P2", 12020, 1.285, 1.071, 1.543, "0.009", nan),
    ("ZTP period premium 2020-2024 (no lc)", "oa_P3", 12020, 1.720, 1.284, 2.304, "<.001", nan),
    ("ZTP last minus first (no lc)", "contrast", 12020, 0.816, 0.441, 1.507, "0.500", nan),
    ("ZTP 2015-19 minus 2003-09 (no lc) [reparametrised]", "oa_negP0", 12020, 0.609, 0.331, 1.122, "0.107", 0.161),
    ("ZTP period premium 2003-2009 (+ lc)", "oa_P0", 12020, 1.643, 1.026, 2.631, "0.040", nan),
    ("ZTP period premium 2010-2014 (+ lc)", "oa_P1", 12020, 1.274, 0.890, 1.824, "0.176", nan),
    ("ZTP period premium 2015-2019 (+ lc)", "oa_P2", 12020, 1.223, 1.014, 1.474, "0.036", nan),
    ("ZTP period premium 2020-2024 (+ lc)", "oa_P3", 12020, 1.605, 1.246, 2.068, "<.001", nan),
    ("ZTP last minus first (+ lc)", "contrast", 12020, 0.977, 0.589, 1.622, "0.926", nan),
    ("ZTP 2015-19 minus 2003-09 (+ lc) [reparametrised]", "oa_negP0", 12020, 0.744, 0.441, 1.258, "0.257", 0.312),
    ("ZTP 2015-19 minus 2003-09, pub<=2019 (no lc)", "oa_negP0", 6640, 0.591, 0.322, 1.086, "0.087", 0.155),
    ("ZTP 2015-19 minus 2003-09, pub<=2019 (+ lc)", "oa_negP0", 6640, 0.733, 0.435, 1.235, "0.231", 0.258),
    ("ZTP 2015-19 minus 2003-09, Gold/Hybrid vs closed (no lc)", "oa_negP0", 10553, 0.830, 0.477, 1.444, "0.494", 0.535),
    ("ZTP OA x Post2013 (no lc) [secondary; window includes 2020-24]", "oa_post", 12020, 0.692, 0.454, 1.053, "0.082", 0.125),
    ("ZTP OA x Post2013 (+ lc) [secondary]", "oa_post", 12020, 0.844, 0.545, 1.308, "0.433", 0.438),
    ("ZTP OA x Post2015 (no lc) [secondary]", "oa_post", 12020, 0.800, 0.512, 1.250, "0.312", 0.354),
    ("ZTP OA x Post2015 (+ lc) [secondary]", "oa_post", 12020, 0.958, 0.631, 1.453, "0.832", 0.846),
    ("ZTP OA x lagged cluster OA share (no lc) [exploratory]", "oa_x_lag", 11596, 1.059, 0.467, 2.404, "0.886", 0.891),
    ("ZTP OA x lagged cluster OA share (+ lc) [exploratory]", "oa_x_lag", 11596, 1.134, 0.494, 2.600, "0.758", 0.728)])
add("step3", "H3", [
    ("ZTP M0 colour only", "c_gold", 11955, 1.330, 1.122, 1.575, "0.002", 0.002),
    ("ZTP M0 colour only", "c_green", 11955, 2.110, 1.786, 2.493, "<.001", 0.001),
    ("ZTP M0 colour only", "c_hybrid", 11955, 1.758, 1.311, 2.357, "<.001", nan),
    ("ZTP M0 colour only", "c_bronze", 11955, 0.974, 0.770, 1.232, "0.818", nan),
    ("ZTP M0 - Green/Gold", "green_minus_gold", 11955, 1.587, 1.219, 2.067, "0.001", nan),
    ("ZTP M1 + lc", "c_gold", 11955, 1.303, 1.104, 1.537, "0.003", 0.007),
    ("ZTP M1 + lc", "c_green", 11955, 1.526, 1.291, 1.805, "<.001", 0.001),
    ("ZTP M1 + lc", "c_hybrid", 11955, 1.187, 0.971, 1.451, "0.092", nan),
    ("ZTP M1 + lc", "c_bronze", 11955, 0.876, 0.703, 1.091, "0.224", nan),
    ("ZTP M1 + lc - Green/Gold", "green_minus_gold", 11955, 1.172, 0.894, 1.536, "0.239", nan),
    ("ZTP M2 + lc + pub-type FE + field tags", "c_gold", 11955, 1.366, 1.144, 1.632, "0.001", 0.004),
    ("ZTP M2 + lc + pub-type FE + field tags", "c_green", 11955, 1.498, 1.254, 1.789, "<.001", 0.001),
    ("ZTP M2 + lc + pub-type FE + field tags", "c_hybrid", 11955, 1.180, 0.952, 1.462, "0.125", nan),
    ("ZTP M2 + lc + pub-type FE + field tags", "c_bronze", 11955, 0.908, 0.727, 1.133, "0.377", nan),
    ("ZTP M2 - Green/Gold", "green_minus_gold", 11955, 1.096, 0.807, 1.489, "0.541", nan),
    ("ZTP M1 Green/Gold, 2003-2014", "green_minus_gold", 2084, 1.353, 0.918, 1.994, "0.120", nan),
    ("ZTP M1 Green/Gold, 2015-2024", "green_minus_gold", 9871, 1.042, 0.827, 1.312, "0.717", nan)])
add("step4", "H3", [
    ("ZTP M0 - Bronze/Gold", "bronze_minus_gold", 11955, 0.732, 0.589, 0.910, "0.007", nan),
    ("ZTP M1 + lc - Bronze/Gold", "bronze_minus_gold", 11955, 0.672, 0.544, 0.831, "<.001", nan),
    ("cluster bootstrap: gap(M0)/gap(M1) ratio (P7)", "gap_reduction", 11955, 1.355, 1.243, 1.489, "0.007", nan)])
add("step4", "H2a", [
    ("ZTP OA x centred lc (RR per 1 log-unit of academic citations)", "oa_x_lc", 12020, 1.146, 1.009, 1.301, "0.037", 0.012),
    ("ZTP OA premium at mean visibility", "oa", 12020, 1.192, 1.016, 1.399, "0.033", nan),
    ("ZTP OA premium at lc 10th pct (lc=1.61)", "oa_at_q", 12020, 0.911, 0.615, 1.349, "0.628", nan),
    ("ZTP OA premium at lc 50th pct (lc=3.64)", "oa_at_q", 12020, 1.200, 1.028, 1.401, "0.023", nan),
    ("ZTP OA premium at lc 90th pct (lc=5.39)", "oa_at_q", 12020, 1.523, 1.330, 1.744, "<.001", nan),
    ("ZTP OA premium, visibility tercile 1 (lowest)", "oa", 4142, 1.079, 0.743, 1.569, "0.677", 0.705),
    ("ZTP OA premium, visibility tercile 2", "oa", 3903, 1.183, 0.940, 1.489, "0.145", 0.187),
    ("ZTP OA premium, visibility tercile 3 (highest)", "oa", 3975, 1.751, 1.478, 2.074, "<.001", 0.001)])
add("step4", "H2b", [
    ("ZTP pub<=2019 OA x year, per 5 years (no lc)", "oa_x_yr", 6640, 0.814, 0.652, 1.017, "0.069", 0.116),
    ("ZTP pub<=2019 OA x year, per 5 years (+ lc)", "oa_x_yr", 6640, 0.908, 0.732, 1.128, "0.369", 0.405)])
add("step5", "M3", [
    ("ZTNB H2a OA x centred lc (alpha at bound)", "oa_x_lc", 12020, 1.185, 1.061, 1.325, "0.004", nan),
    ("ZTNB OA premium at lc 10th pct", "oa_at_q", 12020, 0.985, 0.717, 1.352, "0.921", nan),
    ("ZTNB OA premium at lc 50th pct", "oa_at_q", 12020, 1.391, 1.191, 1.624, "<.001", nan),
    ("ZTNB OA premium at lc 90th pct", "oa_at_q", 12020, 1.873, 1.535, 2.286, "<.001", nan),
    ("ZTNB H2b OA x year per 5y (pub<=2019)", "oa_x_yr", 6640, 0.852, 0.653, 1.110, "0.222", nan),
    ("ZTNB M0", "c_gold", 11955, 1.448, 1.167, 1.797, "0.002", nan),
    ("ZTNB M0", "c_green", 11955, 2.719, 2.149, 3.440, "<.001", nan),
    ("ZTNB M0", "c_hybrid", 11955, 1.870, 1.302, 2.686, "0.002", nan),
    ("ZTNB M0", "c_bronze", 11955, 1.024, 0.639, 1.640, "0.920", nan),
    ("ZTNB M0 - Green/Gold", "green_minus_gold", 11955, 1.877, 1.395, 2.527, "<.001", nan),
    ("ZTNB M0 - Bronze/Gold", "bronze_minus_gold", 11955, 0.707, 0.426, 1.173, "0.171", nan),
    ("ZTNB M1 + lc", "c_gold", 11955, 1.394, 1.144, 1.699, "0.002", nan),
    ("ZTNB M1 + lc", "c_green", 11955, 1.701, 1.362, 2.125, "<.001", nan),
    ("ZTNB M1 + lc", "c_hybrid", 11955, 1.345, 1.010, 1.791, "0.043", nan),
    ("ZTNB M1 + lc", "c_bronze", 11955, 0.808, 0.507, 1.287, "0.355", nan),
    ("ZTNB M1 + lc - Green/Gold", "green_minus_gold", 11955, 1.220, 0.943, 1.578, "0.124", nan),
    ("ZTNB M1 + lc - Bronze/Gold", "bronze_minus_gold", 11955, 0.580, 0.351, 0.959, "0.035", nan),
    ("ZTNB H1 total", "oa", 12020, 1.670, 1.409, 1.980, "<.001", nan),
    ("ZTNB H1 + lc", "oa", 12020, 1.416, 1.204, 1.667, "<.001", nan)])
add("step5", "M5", [
    ("DEDUP ZTP H1 total (no lc)", "oa", 9903, 1.526, 1.342, 1.734, "<.001", 0.001),
    ("DEDUP ZTP H1 + lc", "oa", 9903, 1.329, 1.198, 1.474, "<.001", 0.001),
    ("DEDUP ZTP H2a OA x centred lc", "oa_x_lc", 9903, 1.129, 1.023, 1.247, "0.018", 0.014),
    ("DEDUP ZTP M0 - Green/Gold", "green_minus_gold", 9846, 1.512, 1.193, 1.915, "0.001", nan),
    ("DEDUP ZTP M0 - Bronze/Gold", "bronze_minus_gold", 9846, 0.694, 0.570, 0.847, "<.001", nan),
    ("DEDUP ZTP M1 + lc - Green/Gold", "green_minus_gold", 9846, 1.108, 0.893, 1.375, "0.337", nan),
    ("DEDUP ZTP M1 + lc - Bronze/Gold", "bronze_minus_gold", 9846, 0.639, 0.528, 0.774, "<.001", nan)])
add("step5", "O1", [
    ("ZTP repository copy (all papers, no lc)", "uw_repo", 11905, 1.323, 1.174, 1.490, "<.001", 0.001),
    ("ZTP repository copy (all papers, + lc)", "uw_repo", 11905, 1.190, 1.061, 1.335, "0.005", 0.003),
    ("ZTP repository copy (Lens-OA papers only, no lc)", "uw_repo", 6832, 1.079, 0.902, 1.290, "0.391", 0.378),
    ("ZTP repository copy (Lens-OA papers only, + lc)", "uw_repo", 6832, 1.020, 0.887, 1.174, "0.771", 0.777),
    ("ZTP Gold only: repository copy vs none (+ lc)", "uw_repo", 5028, 0.939, 0.828, 1.065, "0.314", nan),
    ("ZTP M1 Green split by repository version", "c_gold", 11850, 1.288, 1.086, 1.527, "0.005", nan),
    ("ZTP M1 Green split by repository version", "g_oth [CAVEAT: includes 299 Green records with no version info]", 11850, 1.638, 1.248, 2.150, "0.001", nan),
    ("ZTP M1 Green split by repository version", "g_sub (preprint-only repository copy)", 11850, 1.483, 1.217, 1.806, "<.001", nan),
    ("ZTP M1 Green split by repository version", "c_hybrid", 11850, 1.112, 0.895, 1.381, "0.324", nan),
    ("ZTP M1 Green split by repository version", "c_bronze", 11850, 0.842, 0.679, 1.044, "0.111", nan),
    ("ZTP M1: Green(preprint-only)/Gold", "gsub_minus_gold", 11850, 1.152, 0.852, 1.556, "0.343", nan),
    ("ZTP M1: Green(other; see caveat)/Gold", "goth_minus_gold", 11850, 1.272, 0.909, 1.780, "0.152", nan)])
add("step6", "C", [
    ("H1 OA total | SE by institution (G=33)", "oa", 12020, 1.524, 1.309, 1.774, "<.001", nan),
    ("H1 OA total | institution FE, SE by research cluster (G=25)", "oa", 12020, 1.468, 1.321, 1.631, "<.001", nan),
    ("H1 OA + lc | SE by institution (G=33)", "oa", 12020, 1.338, 1.188, 1.508, "<.001", nan),
    ("H1 OA + lc | institution FE, SE by research cluster (G=25)", "oa", 12020, 1.310, 1.190, 1.443, "<.001", nan),
    ("H2a OA x lc_c | SE by institution (G=33)", "oa_x_lc", 12020, 1.146, 1.046, 1.255, "0.005", nan),
    ("H2a OA x lc_c | institution FE, SE by research cluster (G=25)", "oa_x_lc", 12020, 1.156, 1.016, 1.315, "0.029", nan),
    ("H3 Green/Gold M1 | SE by institution (G=33)", "gap", 11955, 1.172, 0.990, 1.386, "0.064", nan),
    ("H3 Green/Gold M1 | institution FE, SE by research cluster (G=25)", "gap", 11955, 1.164, 0.899, 1.507, "0.237", nan)])

df = pd.DataFrame(L, columns=cols)
df.to_csv(R / "primary/ALL_LEDGER.csv", index=False)
for blk, f in [("H1", "primary/h1_ledger.csv"), ("H2", "primary/h2_periods_ledger.csv"), ("H2a", "primary/h2a_visibility_ledger.csv"),
               ("H2b", "primary/h2b_trend_ledger.csv"), ("H3", "primary/h3_colour_ledger.csv"), ("M3", "robustness/ztnb_ledger.csv"),
               ("M5", "robustness/dedup_ledger.csv"), ("O1", "unpaywall/repository_and_green_version_ledger.csv"),
               ("C", "robustness/institution_robustness_ledger.csv")]:
    df[df.block == blk].to_csv(R / f, index=False)

# seven pre-specified / exploratory primary tests
pt = pd.DataFrame([
    ("P1_H1_total", "ZTP OA (no lc)", 1.524, 1.361, 1.706, "<.001", 0.001, "<.001", 0.007),
    ("P2_H1_adj", "ZTP OA + lc", 1.338, 1.211, 1.480, "<.001", 0.001, "<.001", 0.007),
    ("P3_H2_contrast", "2015-19 / 2003-09 OA premium (no lc)", 0.609, 0.331, 1.122, "0.107", 0.161, 0.214, 0.348),
    ("P4_H3_gap", "Green/Gold, M1 (+ lc)", 1.172, 0.894, 1.536, "0.239", nan, 0.239, 0.348),
    ("P5_H2a_visibility", "OA x centred log academic citations (per log-unit)", 1.146, 1.009, 1.301, "0.037", 0.012, 0.024, 0.048),
    ("P6_H2b_trend", "OA x year per 5y, pub<=2019 (no lc)", 0.814, 0.652, 1.017, "0.069", 0.116, 0.116, 0.348),
    ("P7_H3_attenuation", "gap(M0)/gap(M1), cluster bootstrap", 1.355, 1.243, 1.489, "0.007", nan, 0.020, 0.033)],
    columns=["test", "spec", "RR", "RR_lo", "RR_hi", "p_t", "p_boot", "holm_family", "holm_all7"])
pt["family"] = ["step3"] * 4 + ["step4"] * 3
pt["status"] = ["specified in script before ZTP estimation, after exploratory log-scale round"] * 4 + \
    ["exploratory: formulated after inspecting step 3", "post hoc", "exploratory: formulated after inspecting step 3"]
pt.to_csv(R / "primary/primary_tests_P1_P7.csv", index=False)

pd.DataFrame([("H1 total", 1.524, 1.361, 1.706, 1.526, 1.342, 1.734), ("H1 + lc", 1.338, 1.211, 1.480, 1.329, 1.198, 1.474),
              ("H2a interaction", 1.146, 1.009, 1.301, 1.129, 1.023, 1.247), ("H3 Green/Gold M0", 1.587, 1.219, 2.067, 1.512, 1.193, 1.915),
              ("H3 Green/Gold M1", 1.172, 0.894, 1.536, 1.108, 0.893, 1.375)],
             columns=["spec", "rows_RR", "rows_lo", "rows_hi", "papers_RR", "papers_lo", "papers_hi"]
             ).assign(rows_N=[12020, 12020, 12020, 11955, 11955], papers_N=[9903, 9903, 9903, 9846, 9846]
                      ).to_csv(R / "robustness/rows_vs_dedup.csv", index=False)
pd.DataFrame([("H1 total", 1.524, 1.361, 1.706, 1.670, 1.409, 1.980), ("H1 + lc", 1.338, 1.211, 1.480, 1.416, 1.204, 1.667),
              ("H2a interaction", 1.146, 1.009, 1.301, 1.185, 1.061, 1.325), ("H3 Green/Gold M1", 1.172, 0.894, 1.536, 1.220, 0.943, 1.578),
              ("H3 Green/Gold M0", 1.587, 1.219, 2.067, 1.877, 1.395, 2.527)],
             columns=["spec", "ztp_RR", "ztp_lo", "ztp_hi", "ztnb_RR", "ztnb_lo", "ztnb_hi"]).to_csv(R / "robustness/ztp_vs_ztnb.csv", index=False)
pd.DataFrame([("oa", 12020, 22026.5, -14026.6), ("oa+lc", 12020, 22026.5, -13584.4), ("oa+lc_c+oa_x_lc", 12020, 22026.5, -13567.0),
              ("oa+oa_x_yr", 6640, 22026.5, -9805.02), ("colours (M0)", 11955, 22026.5, -13878.9), ("colours + lc (M1)", 11955, 22026.5, -13475.2)],
             columns=["xvars", "n", "alpha", "loglik"]).assign(
    la_at_bound=True, note="alpha = exp(10) upper bound = log-series limit; ZTNB is a direction check only").to_csv(
    R / "robustness/ztnb_dispersion.csv", index=False)
pd.DataFrame([("2003-2024 (main)", 12020, .0614, .0353, .0876, .0548, .0248, .0848), ("1992-2024", 12046, .0621, .0357, .0885, .0548, .0248, .0848),
              ("1992-2025", 12302, .0598, .0338, .0858, .0526, .0228, .0823), ("2003-2025", 12276, .0592, .0333, .0850, .0526, .0228, .0823),
              ("2010-2024", 11537, .0610, .0347, .0872, .0551, .0256, .0846)],
             columns=["window", "n", "oa_beta_adj", "oa_lo", "oa_hi", "goldhybrid_beta_adj", "gh_lo", "gh_hi"]
             ).to_csv(R / "robustness/sample_window_sensitivity_logOLS.csv", index=False)

# sample flow
pd.DataFrame([("raw export rows", 13350), ("empty padding rows", -1000), ("missing patent-citation count", -2), ("= records after cleaning", 12348),
              ("missing publication year", -42), ("published 2026", -4), ("= records with valid year", 12302), ("published before 2003", -26),
              ("published 2025 (partial year)", -256), ("= main analysis sample (2003-2024)", 12020), ("unique papers (Lens ID) in main sample", 9903),
              ("records with unclassified colour ('other')", -65), ("= colour-model sample", 11955), ("unique papers in colour models", 9846)],
             columns=["step", "records"]).to_csv(R / "primary/sample_flow.csv", index=False)

# LOO summary
loo = pd.read_csv(R / "robustness/C3_leave_one_institution_out.csv")
s = loo.groupby("spec").agg(RR_min=("RR", "min"), RR_max=("RR", "max"), n_sig=("p", lambda x: int((x < .05).sum())), n_total=("p", "size")).reset_index()
s.to_csv(R / "robustness/C3_leave_one_institution_out_summary.csv", index=False); print(s)

# Lens vs Unpaywall
src = pd.DataFrame([[147, 107, 16, 3, 22], [93, 4844, 5, 104, 27], [70, 139, 4569, 54, 196], [199, 255, 11, 569, 36], [10, 4, 10, 1, 359], [21, 17, 2, 8, 7]],
                   columns=["bronze", "closed", "gold", "green", "hybrid"],
                   index=pd.Index(["bronze", "closed", "gold", "green", "hybrid", "other"], name="lens_colour (rows) / unpaywall_status (cols)"))
src.to_csv(R / "unpaywall/lens_colour_vs_unpaywall_status.csv")

# Step 1 / Step 2 exploratory
pd.DataFrame([
    ("H1 total (no lc)", "oa", 12020, 0.0730, 0.0456, 0.1004, "<.001", 0.001), ("H1 + lc", "oa", 12020, 0.0614, 0.0353, 0.0876, "<.001", 0.001),
    ("H1 + lc + pub-type FE + field tags", "oa", 12020, 0.0791, 0.0532, 0.1049, "<.001", 0.001),
    ("Gold/Hybrid vs closed, no lc", "oa_pub", 10553, 0.0511, 0.0192, 0.0830, "0.003", 0.004),
    ("Gold/Hybrid vs closed, + lc", "oa_pub", 10553, 0.0548, 0.0248, 0.0848, "<.001", 0.002),
    ("Green vs closed, + lc", "oa_gr", 6209, 0.1285, 0.0781, 0.1788, "<.001", 0.001),
    ("Period premium 2003-09 (no lc)", "oa_P0", 12020, 0.1831, 0.0446, 0.3215, "0.012", nan),
    ("Period premium 2010-14 (no lc)", "oa_P1", 12020, 0.2043, 0.1296, 0.2790, "<.001", nan),
    ("Period premium 2015-19 (no lc)", "oa_P2", 12020, 0.0725, 0.0303, 0.1148, "0.002", nan),
    ("Period premium 2020-24 (no lc)", "oa_P3", 12020, 0.0268, 0.0051, 0.0485, "0.018", nan),
    ("Last minus first (no lc)", "diff", 12020, -0.1563, -0.2929, -0.0196, "0.027", nan),
    ("Last minus first (+ lc)", "diff", 12020, -0.0885, -0.2236, 0.0465, "0.189", nan),
    ("OA x lagged cluster OA share (no lc)", "oa_x_lag", 11596, 0.0034, -0.1159, 0.1227, "0.953", 0.962),
    ("OA x lagged cluster OA share (+ lc)", "oa_x_lag", 11596, -0.0114, -0.1242, 0.1014, "0.837", 0.841),
    ("Green minus Gold M0", "green_minus_gold", 12020, 0.1350, 0.0814, 0.1886, "<.001", nan),
    ("Green minus Gold M1", "green_minus_gold", 12020, 0.0606, 0.0087, 0.1125, "0.024", nan),
    ("Green minus Gold M2", "green_minus_gold", 12020, 0.0457, -0.0075, 0.0989, "0.089", nan),
    ("Decomposition: total", "total", 12020, 0.073, 0.050, 0.101, "", nan),
    ("Decomposition: via academic citations (16% of total)", "indirect", 12020, 0.012, 0.003, 0.025, "", nan)],
    columns=["spec", "term", "n", "beta_log1p", "lo", "hi", "p_t", "p_wcb"]).to_csv(R / "exploratory/step1_log1p_first_round_key_coefficients.csv", index=False)
pd.DataFrame([
    ("2003-2009", 0, 273, 2.663, 1, .505, .495, .297, 3.04), ("2003-2009", 1, 210, 5.971, 2, .419, .581, .410, 4.242),
    ("2010-2014", 0, 797, 2.379, 1, .617, .383, .208, 3.147), ("2010-2014", 1, 819, 3.889, 2, .481, .519, .358, 4.141),
    ("2015-2019", 0, 1766, 1.878, 1, .662, .338, .158, 3.649), ("2015-2019", 1, 2775, 2.404, 1, .588, .412, .234, 3.994),
    ("2020-2024", 0, 2270, 1.241, 1, .821, .179, .041, 3.134), ("2020-2024", 1, 3110, 1.621, 1, .736, .264, .100, 3.495)],
    columns=["period", "oa", "n", "mean_patents", "median_patents", "share_eq1", "share_ge2", "share_ge3", "mean_lc"]
).to_csv(R / "exploratory/step2_descriptives_period_by_oa.csv", index=False)
print("ledger rows:", len(df))
