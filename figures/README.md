# Figures

All figures are journal-style black-and-white, 600 dpi PNG (downscaled to <= 2400 px here) plus vector PDF with editable text (`pdf.fonttype 42`).
File names use **manuscript numbering**; the legacy names used inside the plotting scripts are in the last column.

| Manuscript | File | Content | Produced by | Legacy name | Needs raw data? |
|---|---|---|---|---|---|
| Fig. 1 | `main/Fig1_sample_flow` | sample construction 13,350 -> 12,020 records | `make_figures.py` | Fig2_sample_flow | no |
| Fig. 2 | `main/Fig2_annual_papers_oa_share` | annual records + OA share (Wilson CI) | `make_data_figures.py` | Fig4_annual_papers | **yes** |
| Fig. 3 | `main/Fig3_H1_forest` | H1 OA rate ratios | `make_figures.py` | Fig6_H1_forest | no |
| Fig. 4 | `main/Fig4_H2a_visibility` | OA rate ratio by academic visibility | `make_figures.py` | Fig7_H2a_visibility | no |
| Fig. 5 | `main/Fig5_H2b_periods` | OA rate ratio by period | `make_figures.py` | Fig8_H2b_periods | no |
| Fig. 6 | `main/Fig6_colour_composition` | OA colour composition by period | `make_data_figures.py` | Fig11A_colour_composition | **yes** |
| Fig. 7 | `main/Fig7_academic_citations_by_colour` | cumulative academic citations by colour | `make_data_figures.py` | Fig14A_academic_citations_by_colour | **yes** |
| Fig. 8 | `main/Fig8_H3_colour` | colour rate ratios and Green/Gold, Bronze/Gold contrasts | `make_figures.py` | Fig11_H3_colour | no |
| Fig. A1 | `appendix/FigA1_lens_vs_unpaywall` | Lens x Unpaywall colour agreement | `make_figures.py` | FigN2_lens_unpaywall | no |
| Fig. A2 | `appendix/FigA2_patent_counts` | citing patents per record by OA status and colour | `make_data_figures.py` | FigA1_patent_counts | **yes** |
| Fig. A3 | `appendix/FigA3_umap_clusters` | 2-D UMAP of the 25 archived clusters | `make_data_figures.py` / `make_fig3_umap.py` | Fig3_UMAP_clusters | **yes** (+ embeddings) |
| Fig. A4 | `appendix/FigA4_leave_one_institution_out` | leave-one-institution-out for H1, H2a, H3 | `make_figures.py` | FigN1_leave_one_institution_out | no (uses `results/robustness/C3_*.csv`) |

**Ledger-based figures** (7) take their numbers from the analysis ledger (hard-coded in `make_figures.py` from `FINAL_REPORT.md` / `STEP6_REPORT.md`; Fig. A4 reads the C3 CSV)
and can be rebuilt with `make figures` without any licensed data. **Data-driven figures** (5) are computed from the raw export
(`make figures-all`); copy the resulting files into `figures/` with `bash scripts/figures/collect_figures.sh <dir>`. Run `make check-figures` to see which of the 12 PNGs are present.
