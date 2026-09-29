# Data dictionary (columns read by `scripts/pipeline/`)

The pipeline reads the **processed** file (raw Lens export + `cluster`). Names below are the processed names; `step1/3` `load()` also
accepts `Is Open Access` when `OA` is absent.

| Column | Type | Used for | Notes |
|---|---|---|---|
| `Lens ID` | str | dedup (M5), sample facts | one row per work-institution match, so IDs repeat (10,114 unique of 12,302) |
| `Title` | str | UMAP / cluster terms (figures only) | |
| `DOI` | str | Unpaywall lookup | lower-cased; `https://doi.org/` prefix stripped |
| `Pub_Year` (`Publication Year` in raw) | int | year FE, window | rows with missing year or 2026 are dropped |
| `Citing_Patents` | int >= 1 | **outcome** `y` | count of distinct citing patent documents (not aggregated to families) |
| `Citing_Works` | int | `lc = log(1 + Citing_Works)` | academic citations at extraction; cumulative, possibly post-OA |
| `OA` / `Is Open Access` | 0/1 or bool | **treatment** `oa` | Lens binary OA indicator, measured at extraction, undated |
| `Open Access Colour` | gold/green/hybrid/bronze/blank | H3 colour dummies | blank + `oa=1` -> `other` (65 records, dropped from colour models); closed = reference |
| `Pub_Type` | str | pub-type FE (M2 / H1-c) | |
| `Fields of Study` | `;`/`,`-separated | `ln_fields = log(1 + #tags)` | |
| `Institution` | str | institution robustness (step6), facts | first listed institution is the "primary institution" |
| `cluster` | int (0-24) | cluster FE and inference level | archived assignment (see data/README.md) |

Derived variables: `lp = log(1+Citing_Patents)`, `lc`, `colour`, `c_gold/c_green/c_hybrid/c_bronze/c_other`, `lagshare` (cluster OA share in t-3..t-1, >= 10 papers; retrospective OA coding),
`ge2`, `ge3` (step2), `oa_pub` (Gold/Hybrid vs closed subsample), `inst` (canonicalised institution, step6), `uw_repo` (Unpaywall repository copy, step5).
