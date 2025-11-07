# QuanTa

Finance-grade technical analysis toolkit powered by [Polars](https://www.pola.rs/), [Plotly](https://plotly.com/python/), and [Optuna](https://optuna.org/).

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)
![Poetry](https://img.shields.io/badge/packaging-poetry-informational?logo=poetry)
![Status](https://img.shields.io/badge/status-active-success)

QuanTa centralises everything needed to explore technical indicators, chart market data, and optimise trading strategies. It provides a clean API to fetch OHLCV data, compose sophisticated charts, and run automated hyper-parameter searches for your strategies.

---

## Key Features

- **Fast market data handling** with Polars-based `YahooFinanceClient`.
- **Indicator library** (SMA, EMA, Bollinger Bands, RSI, MACD, ATR, etc.) with a consistent `Indicator` interface.
- **Interactive charting** via Plotly with configurable overlays, subplots, and trade annotations.
- **Strategy optimisation** baked in through Optuna, including constraint management and importance analysis.
- **Ready-to-run notebooks** showcasing optimisation workflows and strategy exploration.
- **Ergonomic CLI** to fetch/memoize Yahoo Finance data (keyed by `symbol-interval`), inspect caches, and plot charts with optional indicators.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher  
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation

```bash
git clone https://github.com/<your-account>/QuanTa.git
cd QuanTa
poetry install
```

Spawn a shell with the environment activated:

```bash
poetry shell
```

---

## Usage Examples

### CLI quick start

The CLI (shipped as `python -m quanta.cli`) wraps the main flows:

1. **Fetch & cache data**

   ```bash
   poetry run python -m quanta.cli fetch AAPL \
       --from 2023-01-01 \
       --to   2023-12-31 \
       --interval 1h
   ```

   - Cached datasets are stored under the key `SYMBOL-interval` by default (`AAPL-1h` above).  
   - Re-running `fetch` for the same key *merges* new rows with the existing cache instead of overwriting it.

2. **Inspect caches**

   ```bash
   poetry run python -m quanta.cli cache --ticker
   ```

   Displays a Rich table with interval, row count, unique trading days, and last update for each cache key.

3. **Plot**

   ```bash
   poetry run python -m quanta.cli plot \
       --cache-key AAPL-1h \
       --symbol-name "AAPL (1h)" \
       --indicator SMA:period=50 --indicator RSI
   ```

   - `--symbol-name / -n` labels the Plotly figure.  
   - Indicators are optional; none are drawn unless you pass `--indicator`.

### Fetch and Plot Market Data

```python
import polars as pl
from quanta.clients.yfinance import YahooFinanceClient
from quanta.clients.chart import ChartClient
from quanta.utils.ta import SMA, RSI, MACD

# 1. Download OHLCV data
client = YahooFinanceClient()
df = client.get_price("AAPL", from_date="2023-01-01", to_date="2024-12-31", interval="1d")

# 2. Prepare indicators and plot
ta_indicators = [SMA(50), SMA(200), RSI(), MACD()]
chart = ChartClient()
chart.plot(df, symbol="AAPL", indicators=ta_indicators)
```

### Run a Strategy Optimisation

```python
import json
from quanta.ta_clients.optimization import OptimizationClient
from quanta.ta_clients.optimization_strategy import OPTIMIZATION_CONFIG

# Optionally load/modify the base configuration
config = json.loads(json.dumps(OPTIMIZATION_CONFIG))
config["optimization"]["n_trials"] = 200

optimizer = OptimizationClient(config)
optimizer.run(symbol="AAPL", from_date="2020-01-01", to_date="2024-12-31")
results = optimizer.to_dataframe()  # analyse trials, best params, importances...
```

---

## Project Structure

```text
QuanTa/
├── quanta/
│   ├── clients/            # Data providers & charting (Yahoo Finance, Plotly)
│   ├── ta_clients/         # Strategy optimisation logic and configs
│   └── utils/              # Indicators, plotting traces, shared utilities
├── notebooks/              # Exploratory notebooks (optimisation, strategy tests)
├── best_config.json        # Example Optuna best-run snapshot
├── optimization_results.csv# Historical optimisation exports
├── pyproject.toml          # Poetry project metadata
└── README.md
```

---

## Notebooks (Clean Examples)

- `notebooks/optim.ipynb` – Curated walk-through of a simple SMA/RSI optimisation, cleaned and ready to run.  
- `notebooks/optim_long_short.ipynb` – Polished long/short exploration showcasing constraint handling.

Both notebooks are maintained as production-ready examples. Launch them in Jupyter or VS Code after installing dependencies:

```bash
poetry run jupyter lab notebooks/
```

---

## Roadmap

- Add unit tests & CI to ensure indicator correctness.
- Expand indicator catalogue (Ichimoku, Heikin Ashi, volume profile, …).
- Provide reusable CLI/REST endpoints for live usage.
- Publish pre-built dashboards and extend documentation.

Feel free to open issues or discussions for feature requests!

---

## Contributing

1. Fork the repository and create your branch (`git checkout -b feature/my-feature`).  
2. Install dependencies (`poetry install`) and keep the code formatted.  
3. Commit with clear messages and open a pull request describing your changes.

---

## Contact

- **Author**: Mileto 
- **LinkedIn**: Share your feedback or showcase what you build with QuanTa – tag the project to spread the word!

If you use QuanTa in your trading research or dashboards, I’d love to hear about it. Happy analysing!
