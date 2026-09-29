#!/usr/bin/env bash
# Full replication on the licensed Lens.org export. Run from the repository root.
#   RAW=data/processed/Transport_CN_Scholarly_Works_with_cluster.csv bash run_all.sh
# Optional: FETCH_UNPAYWALL=1 UNPAYWALL_EMAIL=you@univ.edu  (adds the Unpaywall block; ~10k requests, resumable cache)
set -euo pipefail
RAW="${RAW:-data/processed/Transport_CN_Scholarly_Works_with_cluster.csv}"
[ -f "$RAW" ] || { echo "Missing $RAW - see data/README.md"; exit 1; }
RAW_ABS="$(cd "$(dirname "$RAW")" && pwd)/$(basename "$RAW")"
cd scripts/pipeline
export RV_RAW="$RAW_ABS" RV_BOOT="${RV_BOOT:-999}" RV_BOOT_ATT="${RV_BOOT_ATT:-300}" RV_BOOT_MED="${RV_BOOT_MED:-300}"
OUT="../../outputs"; mkdir -p "$OUT"

echo "[1/5] step1 (exploratory, log-scale first round)"; RV_OUT="$OUT/step1" python3 -u step1_redesign_analysis.py
echo "[2/5] step2 (exploratory, compression / exposure checks)"; RV_OUT="$OUT/step2" python3 -u step2_compression_checks.py
echo "[3/5] step3 (primary plan P1-P4)"; RV_OUT="$OUT/step3" python3 -u step3_primary_plan.py
echo "[4/5] step5 (runs step3 + step4 + ZTNB + dedup + Unpaywall) -> FINAL_REPORT.md"
if [ "${FETCH_UNPAYWALL:-0}" = "1" ]; then export RV_FETCH=1 RV_EMAIL="${UNPAYWALL_EMAIL:?set UNPAYWALL_EMAIL}"; fi
RV_OUT="$OUT/final" python3 -u step5_full_pipeline.py
echo "[5/5] step6 (institution robustness, leave-one-institution-out)"; RV_OUT="$OUT/step6" python3 -u step6_remaining.py
echo "Done. Reports: outputs/final/FINAL_REPORT.md, outputs/step6/STEP6_REPORT.md"
