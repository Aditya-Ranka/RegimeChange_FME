import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ruptures as rpt, json
px = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
r = np.log(px["NIFTY"].dropna()).diff().dropna()
lv = np.log((r.rolling(21).std() * np.sqrt(252)).dropna())
lvz = ((lv - lv.mean()) / lv.std()).values
ev = pd.to_datetime(["2008-09-15", "2011-08-08", "2013-06-03", "2015-08-24", "2016-11-08",
                     "2018-09-21", "2020-03-12", "2022-02-24", "2024-06-04"])
idx = r.index
ep = np.array([idx.searchsorted(d) for d in ev])
def prf(dp, m):
    dp = np.array(dp)
    return (np.mean([np.min(np.abs(ep - d)) <= m for d in dp]),
            np.mean([np.min(np.abs(dp - e)) <= m for e in ep]))
out = []
# vol PELT with penalty tuned to give ~16 breaks (matched to returns PELT)
for pen in [20, 40, 60, 80, 120, 200]:
    bk = rpt.Pelt(model="l2", min_size=20, jump=5).fit(lvz.reshape(-1, 1)).predict(pen=pen)[:-1]
    pos = [idx.searchsorted(lv.index[i]) for i in bk]
    row = {"pen": pen, "n": len(bk)}
    for m in (10, 20, 40):
        p, rc = prf(pos, m) if len(pos) else (0, 0)
        row[f"P{m}"] = round(p, 3); row[f"R{m}"] = round(rc, 3)
    out.append(row)
print(pd.DataFrame(out).to_string())
# random baseline: k uniformly random breaks, 2000 draws
rng = np.random.default_rng(0)
res = []
for k in (16, 105):
    for m in (10, 20, 40):
        P, R = [], []
        for _ in range(2000):
            pos = rng.integers(0, len(idx), k)
            p, rc = prf(pos, m); P.append(p); R.append(rc)
        res.append(dict(k=k, margin=m, precision=round(np.mean(P), 3), recall=round(np.mean(R), 3)))
print(pd.DataFrame(res).to_string())
pd.DataFrame(out).to_csv("data/vol_matched.csv", index=False)
pd.DataFrame(res).to_csv("data/random_baseline.csv", index=False)
