#!/usr/bin/env python3
"""List which of the 12 manuscript figures are present in figures/ (PNG). Exit code 1 if any is missing."""
from pathlib import Path
import sys
F = {"main": ["Fig1_sample_flow", "Fig2_annual_papers_oa_share", "Fig3_H1_forest", "Fig4_H2a_visibility", "Fig5_H2b_periods",
              "Fig6_colour_composition", "Fig7_academic_citations_by_colour", "Fig8_H3_colour"],
     "appendix": ["FigA1_lens_vs_unpaywall", "FigA2_patent_counts", "FigA3_umap_clusters", "FigA4_leave_one_institution_out"]}
missing = 0
for sub, names in F.items():
    for n in names:
        p = Path("figures") / sub / f"{n}.png"
        ok = p.exists(); missing += (not ok)
        print(("OK      " if ok else "MISSING ") + str(p))
print(f"\n{12 - missing}/12 present")
sys.exit(1 if missing else 0)
