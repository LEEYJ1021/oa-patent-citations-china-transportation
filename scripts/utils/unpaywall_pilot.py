#!/usr/bin/env python3
"""
Unpaywall pilot: does `oa_date` carry information on when a paper first became open?

Finding reported in the paper (Section 3.2): for 239 repository locations, oa_date - publication year had mean -0.01
(SD 0.09) and no location was opened more than one year after publication, so oa_date cannot separate deposit at
publication from later deposit. Hence no date variables are used in the analysis.

Usage:  UNPAYWALL_EMAIL=you@university.edu python3 scripts/utils/unpaywall_pilot.py --raw data/raw/Transport_CN_Scholarly_Works.csv --n 300
Note: `oa_date` lives inside each oa_locations[] entry (and first_oa_location), not at the top level of the response.
"""
import os, json, argparse, urllib.request
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--encoding", default="MacRoman"); ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()
EMAIL = os.environ.get("UNPAYWALL_EMAIL")
if not EMAIL:
    raise SystemExit("Set UNPAYWALL_EMAIL")

raw = pd.read_csv(a.raw, encoding=a.encoding, low_memory=False)
d = raw[["DOI", "Publication Year"]].dropna().drop_duplicates("DOI").sample(a.n, random_state=a.seed)
rows = []
for doi, yr in zip(d["DOI"], d["Publication Year"]):
    try:
        j = json.load(urllib.request.urlopen(f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}", timeout=30))
    except Exception:
        continue
    first = (j.get("first_oa_location") or {}).get("oa_date")
    for loc in (j.get("oa_locations") or []):
        rows.append(dict(doi=doi, pub_year=int(yr), status=j.get("oa_status"), host=loc.get("host_type"),
                         version=loc.get("version"), oa_date=loc.get("oa_date"), pub_date=j.get("published_date"),
                         first=(loc.get("oa_date") == first)))
t = pd.DataFrame(rows)
t["oa_year"] = pd.to_datetime(t["oa_date"], errors="coerce").dt.year
t["lag"] = t["oa_year"] - t["pub_year"]
print("locations:", len(t), "| DOIs:", t.doi.nunique())
print("oa_date coverage by host:\n", t.groupby("host")["oa_date"].apply(lambda s: s.notna().mean()).round(2))
print("lag (oa_year - pub_year) by host:\n", t.groupby("host")["lag"].describe().round(2))
print("share opened >1y after publication, by host:\n", t.groupby("host")["lag"].apply(lambda s: (s > 1).mean()).round(2))
rep = t[t.host == "repository"].copy()
rep["same_as_pub"] = pd.to_datetime(rep["oa_date"]).dt.strftime("%Y-%m-%d") == pd.to_datetime(rep["pub_date"]).dt.strftime("%Y-%m-%d")
print("repository oa_date == published_date:", round(rep["same_as_pub"].mean(), 3))
