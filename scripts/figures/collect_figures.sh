#!/usr/bin/env bash
# Copy generated figures into figures/ using MANUSCRIPT numbering (legacy script names in the middle column).
# Usage: bash scripts/figures/collect_figures.sh <dir with generated PNG/PDF>   (default: ./figures_raw)
set -euo pipefail
SRC="${1:-./figures_raw}"
declare -A MAP=(
  [Fig2_sample_flow]="main/Fig1_sample_flow"
  [Fig4_annual_papers]="main/Fig2_annual_papers_oa_share"
  [Fig6_H1_forest]="main/Fig3_H1_forest"
  [Fig7_H2a_visibility]="main/Fig4_H2a_visibility"
  [Fig8_H2b_periods]="main/Fig5_H2b_periods"
  [Fig11A_colour_composition]="main/Fig6_colour_composition"
  [Fig14A_academic_citations_by_colour]="main/Fig7_academic_citations_by_colour"
  [Fig11_H3_colour]="main/Fig8_H3_colour"
  [FigN2_lens_unpaywall]="appendix/FigA1_lens_vs_unpaywall"
  [FigA1_patent_counts]="appendix/FigA2_patent_counts"
  [Fig3_UMAP_clusters]="appendix/FigA3_umap_clusters"
  [FigN1_leave_one_institution_out]="appendix/FigA4_leave_one_institution_out"
)
for k in "${!MAP[@]}"; do
  for ext in png pdf; do
    [ -f "$SRC/$k.$ext" ] && cp "$SRC/$k.$ext" "figures/${MAP[$k]}.$ext" && echo "ok  $k.$ext -> figures/${MAP[$k]}.$ext"
  done
done
