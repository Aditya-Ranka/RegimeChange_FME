import warnings; warnings.filterwarnings("ignore")
import yfinance as yf, pandas as pd
tk = {"NIFTY":"^NSEI","BANK":"^NSEBANK","VIX":"^INDIAVIX","GOLD":"GOLDBEES.NS","LIQ":"LIQUIDBEES.NS"}
out = {}
for k,v in tk.items():
    d = yf.download(v, start="2007-09-01", end="2025-12-31", auto_adjust=True, progress=False)
    s = d["Close"].squeeze() if len(d) else pd.Series(dtype=float)
    out[k] = s
    print(k, v, len(s), s.index.min() if len(s) else None, s.index.max() if len(s) else None)
pd.DataFrame(out).to_csv("data/prices_raw.csv")
