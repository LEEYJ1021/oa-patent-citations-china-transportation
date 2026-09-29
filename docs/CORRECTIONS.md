# Corrections log: what changed relative to the first submission

Discovered while rebuilding the analysis from code. Each item states the old claim, what the code/logs show, and how the revised paper handles it.

| # | First-submission claim | What the code and logs show | Handling in the revision |
|---|---|---|---|
| 1 | 46 records were removed because `Citing_Patents = 0` | The 46 are 42 missing-year rows and 4 dated 2026. The raw export has **no** zero-patent rows | Sample flow (Fig. 1) and Sec. 3.1 rewritten |
| 2 | "1,001 encoding-error rows removed" | 1,000 rows are empty export padding; 2 lack a patent count. Raw file = 13,350 rows (not 13,349) | Sample flow rewritten |
| 3 | A5: 13,349 as a zero-inclusive full sample | There are no zero-patent papers, so A5 could not test the extensive margin; its source code was not found | A5 withdrawn; estimand stated as intensive margin only |
| 4 | Table A4 window split (62 / 12,240 / --) | Actual: 26 before 2003, 12,020 in 2003-2024, 256 in 2025 (total 12,302) | Table A2 replaced; analysis window 2003-2024 |
| 5 | "12,302 papers" | 10,114 unique Lens IDs; unit is the paper-institution pair | Unit stated everywhere; one-row-per-paper re-estimates (M5) show the same conclusions |
| 6 | Table A1 lists 33 institutions incl. a duplicated Beihang | Data have 33 distinct institutions, Beihang appears once (459 records in the main sample); the manuscript list differed from the data | Table A1 rebuilt from data value counts; 985/211 labels dropped as unverified |
| 7 | Variables: cluster / 50% / pre-2013 / N = 1,414 fields | Code uses field level, 30% threshold, 2015 (and a 3-year lagged cluster share in the new analysis) | Old specification not used; new definitions documented in `data/DATA_DICTIONARY.md` |
| 8 | Continuous dose-response `pre_field_oa x Post` as evidence on compression | The model has no OA term, so it cannot measure OA-premium compression; recent-paper truncation enters directly | Not used |
| 9 | Stacked DiD ATT ~ 0.09 with SE = SD(event-time coefficients)/sqrt(n) | That SE is invalid; treatment timing follows individual papers at the 2,719-field level | Designs not used (Sec. 3.4) |
| 10 | Mediation share 32.3% | Not reproducible; recomputed decomposition = 16% (bootstrap CI for the indirect part 0.003-0.025 log points) | Renamed "statistical decomposition", Appendix B |
| 11 | Table B1 mixed first-stage F and beta across specifications | Bartik row: beta 0.116 in the log vs 0.064 in the manuscript; five exclusion checks failed | IV analysis deleted |
| 12 | "Green-Gold gap persists after adjusting for citations" (value axis) | Adjusted ratio 1.17 [0.89, 1.54]; 1.10 with M2; not distinguishable from 1 | "value axis" and policy recommendation removed |
| 13 | Five-year fixed citation window, quality terciles, IPW-TWFE, Heckman, Oster delta, GRF, Bayesian hierarchical, Conley | No generating code located; no patent-citation dates exist in the data | Removed |
| 14 | Effect size 9.4% | Scale change: ZTP rate ratios refer to the latent untruncated rate | Effects reported as rate ratios; log-OLS in Table A3 |

## Pipeline bugs found and fixed during the rebuild

| Bug | Symptom | Fix |
|---|---|---|
| ZTNB reported `1/alpha` and capped alpha at e^4 | alpha printed as 0.02 (really ~50, then at the bound) and convergence warnings | Multi-start over log-alpha up to e^10, Newton polish, alpha reported correctly; result: alpha at the upper bound -> ZTNB is a direction check |
| Unpaywall `oa_date` read from the top level | all `first_oa_year` NaN | Field lives in `oa_locations[]`; pilot then showed it equals the publication date for repository copies -> **all date variables removed** |
| Colour exact-match printed 47.4% | `closed` rows counted as mismatches | Correct agreement from the cross-tab: 88.1% overall, 83.3% among Lens-OA records (`scripts/utils/agreement_from_crosstab.py`) |
| `g_oth` Green split labelled "accepted/published" | included 299 Green records with no version info | Label caveat kept in `results/unpaywall/repository_and_green_version_ledger.csv`; interpret as "other / unknown version" |
| top-1% logit `c_other` = -14.8, p = 1e-210 | complete separation (65 records, none in the top 1%) | `other` dropped from colour models |

## Still open

* Lens query string / extraction date must be archived (Data section).
* Chinese OA policy timeline (CAS/NSFC 2014) should be checked against primary documents.
* Unpaywall statuses are as of the query date; "submittedVersion" can be a default label.
* Hurdle model (O2) was not run (no zero-patent file).
