#!/usr/bin/env python3
"""
Fig. 3 - 2-D UMAP of the research clusters (black-and-white, journal style).

Run from ~/Research/Scientometrics_R1 (needs internet once, for the MiniLM weights):
  pip install sentence-transformers umap-learn scikit-learn --break-system-packages
  python3 make_fig3_umap.py --clustered ./outputs/Transport_CN_Scholarly_Works_with_cluster.csv --out ./figures_data

Method = phase0_reconstruct_cluster.py: title+abstract+fields+keywords -> all-MiniLM-L6-v2 (normalised)
-> PCA(50) -> UMAP (cosine, n_neighbors=15, min_dist=0.1, seed 2025), here with n_components=2 (visual only).
Cluster IDs are the ARCHIVED 'cluster' column; the 2-D layout is regenerated. Points = 2003-2024 records.
Outputs: Fig3_UMAP_clusters.png/.pdf, umap2d_cache.npy, TableA2_cluster_top_terms.csv
Optional: --embeddings emb.npy (pre-computed, same row order as the filtered file) to skip the model download.
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patheffects as pe

ap = argparse.ArgumentParser()
ap.add_argument("--clustered", default="./outputs/Transport_CN_Scholarly_Works_with_cluster.csv")
ap.add_argument("--out", default="./figures_data"); ap.add_argument("--embeddings", default=None)
ap.add_argument("--encoding", default="utf-8-sig")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42, "savefig.dpi": 600})
SEED = 2025

d = pd.read_csv(a.clustered, encoding=a.encoding, low_memory=False); d.columns = d.columns.str.strip()
d["year"] = pd.to_numeric(d["Publication Year"], errors="coerce")
d = d[d["year"].notna() & (d["year"] != 2026)].reset_index(drop=True)          # same order as embedding step
print(f"records loaded: {len(d):,}")

cache = os.path.join(a.out, "umap2d_cache.npy")
if os.path.exists(cache) and len(np.load(cache)) == len(d):
    xy = np.load(cache); print("UMAP cache used")
else:
    if a.embeddings:
        emb = np.load(a.embeddings)
    else:
        from sentence_transformers import SentenceTransformer
        col = lambda c: d[c].fillna("") if c in d.columns else pd.Series("", index=d.index)
        txt = (col("Title") + " " + col("Abstract") + " " + col("Fields of Study") + " " + col("Keywords")).str.strip()
        txt = [t if t else "transport china research" for t in txt]
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
            txt, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    from sklearn.decomposition import PCA; import umap
    emb = PCA(n_components=min(50, emb.shape[1]), random_state=SEED).fit_transform(emb).astype(np.float32)
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb)
    np.save(cache, xy)

m = ((d["year"] >= 2003) & (d["year"] <= 2024)).values
dd, xy = d[m], xy[m]; cl = dd["cluster"].astype(int).values; ids = sorted(set(cl))

fig, ax = plt.subplots(figsize=(6.6, 5.8))
shades = ["0.20", "0.48", "0.68", "0.84"]
for i, c in enumerate(ids):
    k = cl == c; ax.scatter(xy[k, 0], xy[k, 1], s=2.0, c=shades[i % 4], lw=0, rasterized=True)
cent = np.array([[np.median(xy[cl == c, 0]), np.median(xy[cl == c, 1])] for c in ids])
lab = cent.copy(); span = np.ptp(xy, axis=0); rng = np.random.default_rng(0)
for _ in range(200):                                    # simple label repulsion so IDs do not overlap
    moved = False
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            v = (lab[i] - lab[j]) / span; dist = np.hypot(*v)
            if dist < 0.045:
                v = v / dist * 0.006 if dist > 1e-6 else rng.normal(size=2) * 0.006
                lab[i] += v * span; lab[j] -= v * span; moved = True
    if not moved: break
for c, p, q in zip(ids, lab, cent):
    ax.text(*p, f"C{c}", fontsize=7.5, fontweight="bold", ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.125, 0.005, f"N = {len(dd):,} records, {len(ids)} clusters (archived assignment); 2-D layout regenerated for display.\n"
         "Shading alternates only to separate neighbouring clusters; labels sit at cluster medians.", fontsize=6.8, va="bottom")
fig.subplots_adjust(bottom=0.1)
for e in ("png", "pdf"): fig.savefig(os.path.join(a.out, f"Fig3_UMAP_clusters.{e}"), bbox_inches="tight", pad_inches=0.05, facecolor="white")

from sklearn.feature_extraction.text import TfidfVectorizer
docs = [" ".join(dd.loc[cl == c, "Title"].fillna("")) for c in ids]
v = TfidfVectorizer(stop_words="english", min_df=2, max_features=20000); X = v.fit_transform(docs); T = np.array(v.get_feature_names_out())
pd.DataFrame({"cluster": ids, "n": [int((cl == c).sum()) for c in ids],
              "top_terms": [", ".join(T[np.argsort(-X[i].toarray()[0])[:5]]) for i in range(len(ids))]}
             ).to_csv(os.path.join(a.out, "TableA2_cluster_top_terms.csv"), index=False)
print("done ->", a.out)
