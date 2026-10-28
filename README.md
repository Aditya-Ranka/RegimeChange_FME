# Regime Changes in Indian Equity Markets

Empirical study of volatility-regime detection and out-of-sample forecasting for the NIFTY 50. The project compares offline change-point detection (CPD), a Gaussian hidden Markov model (HMM), conventional volatility features, and India VIX in both forecasting and allocation experiments.

## Main findings

- CPD identifies stress-related shifts in NIFTY 50 volatility and produces fewer, penalty-controlled regimes than the HMM.
- In a strictly out-of-sample weekly walk-forward evaluation, CPD and HMM features did not improve next-month volatility forecasts beyond trailing/EWMA volatility.
- Neither CPD nor HMM features improved prediction of a 5% drawdown; India VIX was the strongest useful feature in the tested specification.
- A regime-based allocation reduced drawdown in the historical sample but did not reliably improve risk-adjusted returns.
- The results do **not** provide evidence that regime detection alone is an alpha signal.

The full research report is available at [`report/report.pdf`](report/report.pdf), with a compact presentation reference in [`cheatsheet/cheatsheet.pdf`](cheatsheet/cheatsheet.pdf).

## Repository layout

| Path | Purpose |
| --- | --- |
| `analysis.py` | Exploratory analysis, change-point detection, HMM fitting, validation, and figures |
| `forecast.py` | Walk-forward volatility and drawdown forecasting experiments |
| `supplement.py` | Supplementary sensitivity and validation analyses |
| `data/download.py` | Optional Yahoo Finance data-download script |
| `data/` | Input snapshot and generated tabular results used by the report |
| `figures/` | Exported figures |
| `report/` | LaTeX source and compiled report |
| `cheatsheet/` | LaTeX source and compiled presentation cheatsheet |

## Reproducibility

The scripts expect to be run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To refresh the price snapshot from Yahoo Finance:

```bash
python data/download.py
```

Then run the analyses:

```bash
python analysis.py
python supplement.py
python forecast.py
```

The download step requires an internet connection and uses the following Yahoo Finance symbols:

- `^NSEI` — NIFTY 50
- `^NSEBANK` — NIFTY Bank
- `^INDIAVIX` — India VIX
- `GOLDBEES.NS` — gold ETF proxy
- `LIQUIDBEES.NS` — liquid fund proxy

The committed data files are research artifacts and may differ from a fresh download because market-data providers can revise historical series.

## Methodology

The analysis uses:

- PELT change-point detection on NIFTY 50 return volatility
- A two-state Gaussian HMM as a regime benchmark
- Trailing and exponentially weighted volatility baselines
- India VIX as an observed market-stress feature
- Ridge regression for volatility forecasting
- Logistic regression for drawdown classification
- Walk-forward evaluation with an embargoed training design
- A monthly regime-based allocation backtest with transaction costs

See the report for the exact windows, targets, penalties, benchmarks, and limitations.

## Data and limitations

The data consists of public market-price and volatility series and derived research outputs. It is not investment advice. Historical backtests are not guarantees of future performance. Important limitations include overlapping forecast targets, a single historical path, a small number of severe drawdown episodes, sensitivity to CPD penalties, approximate event dates, proxy instruments, and omitted taxes/slippage beyond the modeled transaction cost.

## License

No license has been selected yet. Until a license is added, all rights are reserved by the authors. The repository is provided for research and educational use.
