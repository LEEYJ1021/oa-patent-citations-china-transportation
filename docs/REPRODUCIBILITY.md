# Reproducibility notes

**Environment.** Python >= 3.10; `pip install -r requirements.txt`. Tested on 3.10 and 3.12 (CI). Estimation uses numpy/scipy/statsmodels only.

**Seeds.** step1 `2025`; step3 `2026` (wild score bootstrap); step4 `4242` (cluster bootstrap); UMAP display `2025`. Bootstrap p-values vary in the third decimal
across seeds and floors at 1/(B+1); the paper reports "<= .001" (B=999) and "p < .01" (B=300).

**Plan hashes.** Each step writes its plan to disk with a SHA-256 *before* estimation (`PRIMARY_ANALYSIS_PLAN.md`, `STEP4_PLAN.md`, `PLAN_STEP5.md`, `PLAN_STEP6.md`).
Hashes in the reported run: step3 `aeb023ce4d94`, step4 `05b455174cb2`, step5 `ce9599713879`. They document the specification, not a registered pre-registration:
the plans were written after an exploratory log-scale round (see the paper's Sec. 3.4).

**Step dependencies** (all in `scripts/pipeline/`, must stay in one folder):

    step1 <- step2
    step3 <- step4 <- step5 <- step6

**Runtime (licensed data, 12k rows).** step3 ~ minutes; step5 ~ tens of minutes (ZTNB + bootstrap); Unpaywall fetch ~ hours (10k DOIs, resumable); step6 leave-one-out ~ tens of minutes.

**Checking your run against the paper.** Compare `outputs/final/FINAL_REPORT.md` with `results/primary/primary_tests_P1_P7.csv` and `results/primary/ALL_LEDGER.csv`.
Point estimates should match to three decimals; bootstrap p-values may differ slightly.

**Known non-determinism.** (i) the archived `cluster` column must be supplied (re-clustering with new library versions gives 20 clusters); (ii) Unpaywall/Lens classifications change over time.

**No licensed data?** `make synthetic smoke` and `make test` run the whole estimation stack on random data and verify that the curated tables in `results/` are internally consistent.
