# Methods in brief (mirrors Section 3 of the paper)

**Estimand.** Association between OA status / OA colour and the *rate* of citing patents among papers that already have >= 1 citing patent.
It is not the probability of a first patent citation and not a causal effect.

**Primary model - zero-truncated Poisson (ZTP).** For record *i* in research cluster *c* and year *t*:

    E[y_ict] = exp(beta*OA_i + gamma*lc_i + alpha_c + lambda_t),   y >= 1,   lc = log(1 + academic citations)

Estimated by maximum likelihood on the truncated likelihood (own Newton implementation with analytic score/Hessian in `step3_primary_plan.py`).
Effects are **rate ratios exp(beta) on the latent (untruncated) rate**; they are not comparable to log-OLS coefficients or to the 9.4% in the first manuscript.
Cluster + year FE compare OA and closed papers of the same year and cluster, which also absorbs paper age (so no offset is used).

**Sensitivity - zero-truncated negative binomial (ZTNB).** Own likelihood with analytic scores (`step5_full_pipeline.py`). The dispersion estimate sits on the
upper bound of the search range (alpha = e^10 = 22,026; log-series limit), so the likelihood is not maximised in the interior. ZTNB is a **direction check only**.

**Inference (G = 25 clusters).**
* cluster-robust (CRV1) sandwich, small-sample factor, `t(G-1)` critical values;
* null-imposed **wild score bootstrap** (Rademacher; Kline & Santos 2012), 999 draws, smallest attainable p = 0.001;
* H3 gap attenuation: cluster (pairs) bootstrap, 300 draws, percentile CI, reported as "p < .01";
* Holm adjustment within family and over all seven primary tests.

**Hypotheses / tests.**

| Test | Spec | Status |
|---|---|---|
| P1, P2 (H1) | OA rate ratio, without / with `lc` | specified in script before ZTP estimation (after an exploratory log-scale round) |
| P3 (H2b endpoint) | premium(2015-19) / premium(2003-09), reparametrised so the contrast has its own bootstrap p | same |
| P4 (H3) | Green/Gold, M1 (+ `lc`) | same |
| P5 (H2a) | OA x centred `lc` | exploratory - formulated after inspecting step 3 |
| P6 (H2b trend) | OA x (year-2011), pub <= 2019, per 5 years | post hoc |
| P7 (H3 attenuation) | gap(M0)/gap(M1), cluster bootstrap | exploratory |

**H3 models.** M0 colours only; M1 + `lc`; M2 + publication type FE + log field-tag count. Closed/unknown is the reference; `other` colour dropped.

**Robustness blocks.** M3 ZTNB; M5 one row per Lens ID; C1-C3 institution clustering / institution FE / leave-one-institution-out (33 institutions);
O1 Unpaywall (agreement, repository copy, Green by repository version); O2 hurdle logit (needs a file with zero-patent papers - not available; **not run**).

**Not estimated, on purpose.** Event-study, stacked DiD, Bartik/IV and continuous dose-response designs: OA status is undated and there is no single
identifiable adoption event, so they would rest on assumptions the data cannot support. Appendix B keeps a Baron-Kenny-style *statistical decomposition*
(16% of the log-OLS OA coefficient runs through academic citations); it is not a mechanism test.
