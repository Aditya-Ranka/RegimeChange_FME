"""Walk-forward forecasting test: do CPD / HMM regime features improve forecasts of
next-21-day realised volatility and of a >5% drawdown, beyond trailing vol, EWMA vol, VIX?"""
import warnings; warnings.filterwarnings("ignore")
import os, json
import numpy as np, pandas as pd, ruptures as rpt
from multiprocessing import Pool
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN, H, MIN_SIZE, MIN_TRAIN, DD_THR = 750, 21, 20, 150, -0.05
RIDGE_ALPHA, LOGIT_C, BLOCK, NBOOT = 1.0, 1.0, 26, 2000

px = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
r = np.log(px["NIFTY"].dropna()).diff().dropna()
vol21 = r.rolling(21).std() * np.sqrt(252)
ewma = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean() * 252)   # RiskMetrics lambda=0.94
vix = px["VIX"].reindex(r.index).ffill()
vol21_l, ewma_l, vix_l = vol21.shift(1), ewma.shift(1), vix.shift(1)   # known strictly before day d


def fit_hmm(x, seeds=3):
    best, bs = None, -np.inf
    for s in range(seeds):
        m = GaussianHMM(2, covariance_type="full", n_iter=200, random_state=s).fit(x)
        sc = m.score(x)
        if sc > bs: best, bs = m, sc
    return best


def row(d):
    hist = r[r.index < d].iloc[-WIN:]
    hz = ((hist - hist.mean()) / hist.std()).values.reshape(-1, 1)
    bks = rpt.Pelt(model="normal", min_size=MIN_SIZE, jump=5).fit(hz).predict(pen=3 * np.log(len(hz)))[:-1]
    last = bks[-1] if len(bks) else 0
    segvol = hist.iloc[last:].std() * np.sqrt(252)
    med = vol21[vol21.index < d].iloc[-WIN:].median()
    m = fit_hmm(hz)
    hs = int(np.argmax([m.covars_[k].ravel()[0] for k in range(2)]))
    hprob = float(m.predict_proba(hz)[-1, hs])
    fut = r[r.index >= d].iloc[:H]
    if len(fut) < H: return None
    cum = np.expm1(fut.cumsum())
    return dict(date=d, tend=fut.index[-1], v21=vol21_l.loc[d], ew=ewma_l.loc[d], vix=vix_l.loc[d],
                cpd_ratio=segvol / med, cpd_flag=int(segvol > 1.3 * med), days_since=len(hist) - last,
                hmm_prob=hprob, n_breaks=len(bks), y_vol=np.log(fut.std() * np.sqrt(252)),
                y_dd=int(cum.min() <= DD_THR), fut_vol=fut.std() * np.sqrt(252), fut_mdd=cum.min())


if __name__ == "__main__":
    if not os.path.exists("data/features.csv"):
        idx = r.index[WIN:]
        wk = pd.Series(idx, index=idx).groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).head(1)
        with Pool(6) as p:
            rows = [x for x in p.map(row, list(wk.values)) if x is not None]
        pd.DataFrame(rows).to_csv("data/features.csv", index=False)
    df = pd.read_csv("data/features.csv", parse_dates=["date", "tend"]).dropna().reset_index(drop=True)
    df["lv21"], df["lew"], df["lvix"] = np.log(df.v21), np.log(df.ew), np.log(df.vix)
    df["lcpd"] = np.log(df.cpd_ratio)
    print("samples", len(df), df.date.min().date(), df.date.max().date(), "dd base rate", df.y_dd.mean())

    FS = {"F0 trailing+EWMA": ["lv21", "lew"],
          "F1 +CPD": ["lv21", "lew", "lcpd", "cpd_flag"],
          "F2 +HMM": ["lv21", "lew", "hmm_prob"],
          "F3 +VIX": ["lv21", "lew", "lvix"],
          "F4 +VIX+CPD+HMM": ["lv21", "lew", "lvix", "lcpd", "cpd_flag", "hmm_prob"],
          "VIX only": ["lvix"], "CPD+HMM only": ["lcpd", "cpd_flag", "hmm_prob"]}
    n = len(df)
    pv = {k: np.full(n, np.nan) for k in FS}; pd_ = {k: np.full(n, np.nan) for k in FS}
    pmean = np.full(n, np.nan); pbase = np.full(n, np.nan)
    for i in range(n):
        tr = df[df.tend < df.date.iloc[i]]            # embargo: target window fully observed before date i
        if len(tr) < MIN_TRAIN: continue
        pmean[i] = tr.y_vol.mean(); pbase[i] = tr.y_dd.mean()
        for k, cols in FS.items():
            sc = StandardScaler().fit(tr[cols])
            Xt, Xi = sc.transform(tr[cols]), sc.transform(df[cols].iloc[[i]])
            pv[k][i] = Ridge(alpha=RIDGE_ALPHA).fit(Xt, tr.y_vol).predict(Xi)[0]
            if tr.y_dd.nunique() == 2:
                pd_[k][i] = LogisticRegression(C=LOGIT_C, max_iter=1000).fit(Xt, tr.y_dd).predict_proba(Xi)[0, 1]
            else:
                pd_[k][i] = tr.y_dd.mean()
    ok = ~np.isnan(pmean)
    oos = df[ok].reset_index(drop=True)
    y = oos.y_vol.values; yd = oos.y_dd.values
    PV = {k: v[ok] for k, v in pv.items()}; PD = {k: v[ok] for k, v in pd_.items()}
    print("OOS samples", len(oos), oos.date.min().date(), oos.date.max().date(), "OOS dd base", yd.mean(), "positives", yd.sum())

    def dm(e_base, e_mod, L=H - 1):
        d = e_base ** 2 - e_mod ** 2
        T = len(d); dc = d - d.mean()
        lrv = dc @ dc / T + 2 * sum((1 - l / (L + 1)) * (dc[l:] @ dc[:-l]) / T for l in range(1, L + 1))
        stat = d.mean() / np.sqrt(lrv / T)
        from scipy.stats import norm
        return float(stat), float(2 * (1 - norm.cdf(abs(stat))))

    sse_mean = np.sum((y - pmean[ok]) ** 2); e0 = y - PV["F0 trailing+EWMA"]
    sse0 = np.sum(e0 ** 2)
    vrows = []
    naive = y - oos.lv21.values
    vrows.append(dict(model="Expanding mean", RMSE=np.sqrt(np.mean((y - pmean[ok]) ** 2)), MAE=np.mean(np.abs(y - pmean[ok])),
                      R2_vs_mean=0.0, R2_vs_F0=np.nan, DM_stat=np.nan, DM_p=np.nan))
    vrows.append(dict(model="Naive (last 21d vol)", RMSE=np.sqrt(np.mean(naive ** 2)), MAE=np.mean(np.abs(naive)),
                      R2_vs_mean=1 - np.sum(naive ** 2) / sse_mean, R2_vs_F0=1 - np.sum(naive ** 2) / sse0,
                      DM_stat=dm(e0, naive)[0], DM_p=dm(e0, naive)[1]))
    for k in FS:
        e = y - PV[k]; st, p = dm(e0, e) if k != "F0 trailing+EWMA" else (np.nan, np.nan)
        vrows.append(dict(model=k, RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                          R2_vs_mean=1 - np.sum(e ** 2) / sse_mean, R2_vs_F0=1 - np.sum(e ** 2) / sse0, DM_stat=st, DM_p=p))
    vt = pd.DataFrame(vrows).round(4); print(vt.to_string()); vt.to_csv("data/forecast_vol.csv", index=False)

    rng = np.random.default_rng(0)
    N = len(oos); nb = int(np.ceil(N / BLOCK))
    def boot_auc_diff(pa, pb):
        ds = []
        for _ in range(NBOOT):
            st_ = rng.integers(0, N - BLOCK + 1, nb)
            ix = np.concatenate([np.arange(s, s + BLOCK) for s in st_])[:N]
            if len(np.unique(yd[ix])) < 2: continue
            ds.append(roc_auc_score(yd[ix], pa[ix]) - roc_auc_score(yd[ix], pb[ix]))
        return np.percentile(ds, [2.5, 97.5])
    drows = []
    for k in FS:
        p = np.clip(PD[k], 1e-6, 1 - 1e-6)
        ci = boot_auc_diff(p, PD["F0 trailing+EWMA"]) if k != "F0 trailing+EWMA" else (np.nan, np.nan)
        drows.append(dict(model=k, AUC=roc_auc_score(yd, p), Brier=brier_score_loss(yd, p), LogLoss=log_loss(yd, p),
                          dAUC_vs_F0_lo=ci[0], dAUC_vs_F0_hi=ci[1]))
    pb = np.clip(pbase[ok], 1e-6, 1 - 1e-6)
    drows.append(dict(model="Base rate only", AUC=0.5, Brier=brier_score_loss(yd, pb), LogLoss=log_loss(yd, pb),
                      dAUC_vs_F0_lo=np.nan, dAUC_vs_F0_hi=np.nan))
    dt = pd.DataFrame(drows).round(4); print(dt.to_string()); dt.to_csv("data/forecast_dd.csv", index=False)

    cond = []
    for name, mask in [("CPD flag = 1", oos.cpd_flag == 1), ("CPD flag = 0", oos.cpd_flag == 0),
                       ("HMM P(stress) > 0.5", oos.hmm_prob > 0.5), ("HMM P(stress) <= 0.5", oos.hmm_prob <= 0.5),
                       ("India VIX > 25", oos.vix > 25), ("India VIX <= 25", oos.vix <= 25)]:
        cond.append(dict(condition=name, n=int(mask.sum()), mean_next21_vol=oos.fut_vol[mask].mean(),
                         freq_dd5=oos.y_dd[mask].mean(), mean_mdd=oos.fut_mdd[mask].mean()))
    ct = pd.DataFrame(cond).round(4); print(ct.to_string()); ct.to_csv("data/forecast_cond.csv", index=False)

    cols = FS["F4 +VIX+CPD+HMM"]; sc = StandardScaler().fit(df[cols]); Xa = sc.transform(df[cols])
    rg = Ridge(alpha=RIDGE_ALPHA).fit(Xa, df.y_vol); lg = LogisticRegression(C=LOGIT_C, max_iter=1000).fit(Xa, df.y_dd)
    coef = pd.DataFrame(dict(feature=cols, ridge_vol=rg.coef_, logit_dd=lg.coef_[0])).round(4)
    print(coef.to_string()); coef.to_csv("data/forecast_coef.csv", index=False)

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax[0].plot(oos.date, np.exp(y) * 100, "k", lw=0.8, label="Realised (next 21d)")
    ax[0].plot(oos.date, np.exp(PV["F0 trailing+EWMA"]) * 100, lw=0.7, label="F0 baseline")
    ax[0].plot(oos.date, np.exp(PV["F4 +VIX+CPD+HMM"]) * 100, lw=0.7, label="F4 all features")
    ax[0].set_ylabel("Ann. vol (%)"); ax[0].legend(fontsize=6); ax[0].set_title("Out-of-sample volatility forecasts")
    for k, c in [("F0 trailing+EWMA", "tab:blue"), ("F4 +VIX+CPD+HMM", "tab:orange"), ("CPD+HMM only", "tab:green"), ("VIX only", "tab:red")]:
        f, t, _ = roc_curve(yd, PD[k]); ax[1].plot(f, t, c=c, lw=0.9, label=k)
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.5); ax[1].legend(fontsize=5.5); ax[1].set_title("ROC: 21d drawdown > 5%")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); plt.tight_layout(); plt.savefig("figures/forecast.pdf"); plt.close()

    pd.DataFrame(dict(date=oos.date, y_vol=y, y_dd=yd, **{f"pv_{i}": PV[k] for i, k in enumerate(FS)},
                      **{f"pd_{i}": PD[k] for i, k in enumerate(FS)})).to_csv("data/forecast_oos.csv", index=False)
    json.dump(dict(n=len(df), n_oos=len(oos), oos_start=str(oos.date.min().date()), oos_end=str(oos.date.max().date()),
                   dd_base_oos=float(yd.mean()), dd_pos=int(yd.sum()), dd_base_all=float(df.y_dd.mean()),
                   first_sample=str(df.date.min().date())), open("data/forecast_summary.json", "w"), indent=1)
