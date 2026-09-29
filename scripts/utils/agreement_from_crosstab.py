#!/usr/bin/env python3
"""Lens vs Unpaywall colour agreement from results/unpaywall/lens_colour_vs_unpaywall_status.csv.

This is the CORRECT agreement calculation. The step5 pipeline printed 47.4% in one run because its exact-match check
scored Lens 'closed' as a mismatch against Unpaywall 'closed'; do not quote that number.
Expected output: 88.1% overall (10,488/11,905), 83.3% among Lens-OA colours (5,644/6,777)."""
import pandas as pd, sys
p = sys.argv[1] if len(sys.argv) > 1 else "results/unpaywall/lens_colour_vs_unpaywall_status.csv"
ct = pd.read_csv(p, index_col=0)
cols = ["bronze", "closed", "gold", "green", "hybrid"]
diag = sum(ct.loc[c, c] for c in cols)
print(f"all matched records incl. 'other': {diag:,}/{ct.values.sum():,} = {diag/ct.values.sum():.1%}")
oa = ["bronze", "gold", "green", "hybrid"]
d2 = sum(ct.loc[c, c] for c in oa); n2 = ct.loc[oa].values.sum()
print(f"Lens-OA colour records: {d2:,}/{n2:,} = {d2/n2:.1%}")
for c in cols:
    print(f"  Lens {c:7s}: {ct.loc[c, c]:>5,}/{ct.loc[c].sum():>5,} = {ct.loc[c, c]/ct.loc[c].sum():.0%}")
lens_oa = ct.drop(index=["closed", "other"]).sum(axis=1)
print("binary-OA agreement is reported by the pipeline (93.7%); it needs record-level data, not the crosstab.")
