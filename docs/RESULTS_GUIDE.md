# Guide to `results/`

| File | What it is | Paper |
|---|---|---|
| `primary/primary_tests_P1_P7.csv` | the seven primary tests with Holm p (family, all-7) and status | Tables 2-5 text, Sec. 3.4 |
| `primary/ALL_LEDGER.csv` | every reported coefficient (source step, block, spec, RR, CI, p, bootstrap p) | all |
| `primary/h1_ledger.csv`, `h2_periods_ledger.csv`, `h2a_visibility_ledger.csv`, `h2b_trend_ledger.csv`, `h3_colour_ledger.csv` | ledger split by hypothesis | Tables 2, 3, 5 |
| `primary/sample_flow.csv` | 13,350 -> 12,020 records; 9,903 papers | Fig. 1, Sec. 3.1 |
| `robustness/ztp_vs_ztnb.csv`, `ztnb_ledger.csv`, `ztnb_dispersion.csv` | ZTNB direction check; alpha at bound | Sec. 3.4 |
| `robustness/rows_vs_dedup.csv`, `dedup_ledger.csv` | paper-level (one row per Lens ID) re-estimates | Sec. 4.4 |
| `robustness/institution_robustness_ledger.csv` | SE by institution; institution FE | Sec. 4.4 |
| `robustness/C3_leave_one_institution_out.csv` (+ `_summary`) | 33 leave-one-institution-out fits x 3 specs | Fig. A4 |
| `robustness/sample_window_sensitivity_logOLS.csv` | 1992/2003/2025 window sensitivity | Table A2 |
| `unpaywall/lens_colour_vs_unpaywall_status.csv` | Lens x Unpaywall colour cross-tab (11,905 matched) | Fig. A1 |
| `unpaywall/repository_and_green_version_ledger.csv` | repository copy, Green by version | Sec. 4.4 |
| `exploratory/*` | first-round log(1+y) OLS and period descriptives (not the paper's main models) | Table A3 |

Columns of the ledgers: `RR` = exp(beta) on the latent rate (for contrasts, a ratio of rate ratios); `RR_lo/RR_hi` t(G-1) 95% CI; `p` = t(G-1) p-value;
`p_wcb` = wild-score-bootstrap p (where computed). For the log-OLS files, `beta_log1p` is a log-scale coefficient.
Interpretation rules used in the paper: non-significant means "not distinguishable from zero/one", never "no effect"; bootstrap p is floored at 1/(B+1).
