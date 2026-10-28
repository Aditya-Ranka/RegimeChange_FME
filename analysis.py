import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ruptures as rpt, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

np.random.seed(0)
MIN_SIZE, WIN, COST = 20, 750, 0.0010
px = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
nifty = px["NIFTY"].dropna()
r = np.log(nifty).diff().dropna()
vol21 = r.rolling(21).std() * np.sqrt(252)
lv = np.log(vol21.dropna())
thr_full = vol21.median() * 1.3
print("n", len(r), "kurt", r.kurt(), "median vol", vol21.median())

def pelt_normal(x, pen):
    algo = rpt.Pelt(model="normal", min_size=MIN_SIZE, jump=5).fit(x.reshape(-1, 1))
    return algo.predict(pen=pen)

def pelt_l2(x, pen):
    algo = rpt.Pelt(model="l2", min_size=MIN_SIZE, jump=5).fit(x.reshape(-1, 1))
    return algo.predict(pen=pen)

rz = (r.values - r.values.mean()) / r.values.std()
lvz = (lv.values - lv.values.mean()) / lv.values.std()
n = len(rz)
pens = [5, 10, 20, 40, 80, 160]
sweep = []
for p in pens:
    sweep.append((p, len(pelt_normal(rz, p)) - 1, len(pelt_l2(lvz, p / 20.0)) - 1))
print("sweep (pen, #bk returns, #bk logvol[pen/20])", sweep)

PEN_R = 3 * np.log(n)  # BIC-style for 2 params (mean,var)
bk_r = pelt_normal(rz, PEN_R)[:-1]
bk_v = pelt_l2(lvz, 2.0)[:-1]
bs = rpt.Binseg(model="l2", min_size=MIN_SIZE, jump=5).fit(lvz.reshape(-1, 1)).predict(pen=2.0)[:-1]
dates_r = [r.index[i] for i in bk_r]
dates_v = [lv.index[i] for i in bk_v]
dates_b = [lv.index[i] for i in bs]
print("PEN_R", PEN_R, "returns bks", len(dates_r), "logvol bks", len(dates_v), "binseg", len(dates_b))

# regimes from CPD (full-sample, illustrative)
def seg_regime(idx, bks, series_r):
    lab = pd.Series(0, index=idx)
    edges = [0] + list(bks) + [len(idx)]
    for a, b in zip(edges[:-1], edges[1:]):
        v = series_r.iloc[a:b].std() * np.sqrt(252)
        lab.iloc[a:b] = int(v > thr_full)
    return lab
reg_cpd = seg_regime(r.index, bk_r, r)

# HMM full-sample
def fit_hmm(x, seeds=5):
    best, bs_ = None, -np.inf
    for s in range(seeds):
        m = GaussianHMM(2, covariance_type="full", n_iter=200, random_state=s).fit(x)
        sc = m.score(x)
        if sc > bs_: best, bs_ = m, sc
    return best
hm = fit_hmm(rz.reshape(-1, 1))
st = hm.predict(rz.reshape(-1, 1))
hi = int(np.argmax([hm.covars_[k].ravel()[0] for k in range(2)]))
reg_hmm = pd.Series((st == hi).astype(int), index=r.index)
agree = (reg_hmm == reg_cpd).mean()
print("HMM stress share", reg_hmm.mean(), "CPD stress share", reg_cpd.mean(), "agreement", agree)
hmm_sw = int((reg_hmm.diff().abs() > 0).sum())
hmm_dates = list(reg_hmm.index[reg_hmm.diff().abs() > 0])

# events
events = {
 "GFC/Lehman": "2008-09-15", "Euro crisis": "2011-08-08", "Taper tantrum": "2013-06-03",
 "China deval.": "2015-08-24", "Demonetisation": "2016-11-08", "IL&FS": "2018-09-21",
 "COVID crash": "2020-03-12", "Russia-Ukraine/Fed": "2022-02-24", "Election result": "2024-06-04"}
ev = {k: pd.Timestamp(v) for k, v in events.items()}
idx = r.index
def tpos(d): return idx.searchsorted(d)
def prf(dets, margin):
    dp = np.array([tpos(d) for d in dets]); ep = np.array([tpos(d) for d in ev.values()])
    if len(dp) == 0: return 0, 0, np.nan, 0
    rec = np.mean([np.min(np.abs(dp - e)) <= margin for e in ep])
    prec = np.mean([np.min(np.abs(ep - d)) <= margin for d in dp])
    lag = np.mean([np.min(np.abs(dp - e)) for e in ep])
    return prec, rec, lag, len(dp)
tab = []
for name, d in [("PELT-Normal (returns)", dates_r), ("PELT-L2 (log-vol)", dates_v),
                ("BinSeg-L2 (log-vol)", dates_b), ("HMM-2 (returns)", hmm_dates)]:
    for m in (10, 20, 40):
        p, rc, lg, k = prf(d, m)
        tab.append(dict(method=name, margin=m, n=k, precision=round(p, 3), recall=round(rc, 3), mean_dist=round(lg, 1)))
tab = pd.DataFrame(tab); print(tab.to_string())
tab.to_csv("data/event_validation.csv", index=False)
per_event = {k: min([abs((d - v).days) for d in dates_r + dates_v]) for k, v in ev.items()}
print("per-event nearest CPD break (days):", per_event)

# figures
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].semilogy(nifty.index, nifty.values, "k", lw=0.8)
edges = [0] + list(bk_r) + [len(r)]
for a, b in zip(edges[:-1], edges[1:]):
    if reg_cpd.iloc[a]: ax[0].axvspan(r.index[a], r.index[b - 1], color="tab:red", alpha=0.2)
for d in dates_r: ax[0].axvline(d, color="tab:blue", lw=0.5)
for k, d in ev.items(): ax[0].axvline(d, color="tab:green", ls="--", lw=0.8)
ax[0].set_ylabel("NIFTY 50 (log)"); ax[0].set_title("PELT breaks (blue), stress regimes (red), events (green dashed)")
ax[1].plot(vol21.index, vol21.values, "k", lw=0.6, label="21d realised vol")
for d in dates_v: ax[1].axvline(d, color="tab:orange", lw=0.6)
ax[1].axhline(thr_full, color="r", ls=":", label="stress threshold"); ax[1].legend(loc="upper right")
ax[1].set_ylabel("Ann. vol"); plt.tight_layout(); plt.savefig("figures/cpd_overview.pdf"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7, 2.2))
ax[0].plot(nifty.index, np.log(nifty.values), lw=0.7); ax[0].set_title("Log price")
ax[1].plot(r.index, r.values, lw=0.4); ax[1].set_title("Daily log returns")
ax[2].hist(r.values, bins=100, log=True); ax[2].set_title("Return histogram (log y)")
plt.tight_layout(); plt.savefig("figures/eda.pdf"); plt.close()

plt.figure(figsize=(3.4, 2.3))
plt.plot([s[0] for s in sweep], [s[1] for s in sweep], "o-", label="returns (Normal)")
plt.plot([s[0] for s in sweep], [s[2] for s in sweep], "s-", label="log-vol (L2, pen/20)")
plt.xscale("log"); plt.xlabel("Penalty"); plt.ylabel("# breakpoints"); plt.legend(); plt.tight_layout()
plt.savefig("figures/penalty_sweep.pdf"); plt.close()

# ---------------- walk-forward backtest ----------------
assets = px[["NIFTY", "GOLD", "LIQ"]].dropna()
ar = np.log(assets).diff().dropna()
ar = ar[ar.index >= "2009-01-02"]
ar = ar.clip(-0.3, 0.3)  # guard against bad ticks
rb_dates = ar.groupby([ar.index.year, ar.index.month]).head(1).index
start = pd.Timestamp("2011-01-01")
rb_dates = [d for d in rb_dates if d >= start]
cpd_sig = pd.Series(np.nan, index=ar.index); hmm_sig = cpd_sig.copy()
vix = px["VIX"].reindex(ar.index).ffill()
for d in rb_dates:
    hist = r[r.index < d].iloc[-WIN:]
    hz = ((hist - hist.mean()) / hist.std()).values
    bks = pelt_normal(hz, 3 * np.log(len(hz)))[:-1]
    last = bks[-1] if len(bks) else 0
    v = hist.iloc[last:].std() * np.sqrt(252)
    thr = vol21[vol21.index < d].iloc[-WIN:].median() * 1.3
    cpd_sig.loc[d] = int(v > thr)
    m = fit_hmm(hz.reshape(-1, 1), seeds=3)
    hs = int(np.argmax([m.covars_[k].ravel()[0] for k in range(2)]))
    hmm_sig.loc[d] = int(m.predict(hz.reshape(-1, 1))[-1] == hs)
cpd_sig = cpd_sig.ffill(); hmm_sig = hmm_sig.ffill()
vix_sig = (vix > 25).astype(int).where(cpd_sig.notna())

def run(sig):
    s = sig.shift(1)  # one-day lag: no look-ahead
    s = s[s.notna()]
    sub = ar.loc[s.index]
    w = pd.DataFrame({"NIFTY": 1 - s, "GOLD": 0.5 * s, "LIQ": 0.5 * s})
    turn = w.diff().abs().sum(axis=1).fillna(0)
    pr = (w * (np.exp(sub) - 1)).sum(axis=1) - turn * COST
    return pr, int((s.diff().abs() > 0).sum()), turn.sum()

strats = {}
base = ar.loc[cpd_sig.dropna().index]
strats["Buy&Hold NIFTY"] = (np.exp(base["NIFTY"]) - 1, 0, 0)
strats["Static 60/40 NIFTY/Gold"] = ((0.6 * (np.exp(base["NIFTY"]) - 1) + 0.4 * (np.exp(base["GOLD"]) - 1)), 0, 0)
strats["CPD regime (walk-fwd)"] = run(cpd_sig)
strats["HMM regime (walk-fwd)"] = run(hmm_sig)
strats["VIX>25 rule"] = run(vix_sig)
rf = (np.exp(ar["LIQ"].loc[base.index]) - 1)
rows = []; curves = {}
for k, (pr, sw, to) in strats.items():
    pr = pr.dropna(); eq = (1 + pr).cumprod(); curves[k] = eq
    yrs = len(pr) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = pr.std() * np.sqrt(252)
    sh = (pr - rf.reindex(pr.index)).mean() / pr.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    rows.append(dict(strategy=k, CAGR=round(cagr * 100, 2), Vol=round(vol * 100, 2), Sharpe=round(sh, 2),
                     MaxDD=round(dd * 100, 1), Calmar=round(cagr / abs(dd), 2), Switches=sw))
res = pd.DataFrame(rows); print(res.to_string()); res.to_csv("data/backtest.csv", index=False)
print("period", base.index[0], base.index[-1])

fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
for k, eq in curves.items():
    ax[0].plot(eq.index, eq.values, lw=0.8, label=k)
    ax[1].plot(eq.index, (eq / eq.cummax() - 1).values, lw=0.6)
ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1"); ax[0].legend(fontsize=5.5)
ax[1].set_ylabel("Drawdown"); plt.tight_layout(); plt.savefig("figures/backtest.pdf"); plt.close()

json.dump(dict(n=n, kurt=float(r.kurt()), pen_r=float(PEN_R), n_r=len(dates_r), n_v=len(dates_v), n_b=len(dates_b),
               hmm_sw=hmm_sw, agree=float(agree), stress_cpd=float(reg_cpd.mean()), stress_hmm=float(reg_hmm.mean()),
               per_event=per_event, bk_r=[str(d.date()) for d in dates_r], bk_v=[str(d.date()) for d in dates_v],
               sweep=sweep, thr=float(thr_full), medvol=float(vol21.median()),
               bt_start=str(base.index[0].date()), bt_end=str(base.index[-1].date())), open("data/summary.json", "w"), indent=1)
