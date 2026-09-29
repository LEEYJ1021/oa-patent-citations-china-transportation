#!/usr/bin/env python3
"""Generate a SYNTHETIC dataset with the same column layout as the Lens export + `cluster` column.
For smoke-testing the pipeline only. Values are random; results carry no scientific meaning.
Usage: python3 tests/make_synthetic_data.py data/synthetic/synthetic_with_cluster.csv [n_rows]"""
import sys, numpy as np, pandas as pd
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
rng = np.random.default_rng(0)
year = rng.integers(2003, 2025, n)
oa = rng.binomial(1, 0.5, n)
colour = np.where(oa == 0, "", rng.choice(["gold", "green", "hybrid", "bronze"], n, p=[.6, .2, .1, .1]))
lam = 0.6 + 0.3 * oa
y = 1 + rng.poisson(lam * rng.gamma(2, 0.5, n))
inst = rng.choice([f"Institution {i}" for i in range(33)], n)
ids = np.arange(n) % int(n * 0.82)
d = pd.DataFrame({
    "Lens ID": [f"L{i:06d}" for i in ids], "Title": [f"paper {i}" for i in ids], "DOI": [f"10.1000/syn.{i}" for i in ids],
    "Pub_Year": year, "Publication Year": year, "Citing_Patents": y, "Citing_Works": rng.poisson(20, n),
    "OA": oa, "Is Open Access": oa.astype(bool), "Open Access Colour": colour, "Pub_Type": rng.choice(["journal article", "proceedings article"], n),
    "Fields of Study": [";".join(rng.choice(list("abcdefgh"), rng.integers(1, 5))) for _ in range(n)],
    "Institution": inst, "cluster": rng.integers(0, 25, n)})
import os; os.makedirs(os.path.dirname(out), exist_ok=True); d.to_csv(out, index=False); print("wrote", out, d.shape)
