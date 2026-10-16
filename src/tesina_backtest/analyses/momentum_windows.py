"""Análisis 2: robustez de la ventana de momentum con la regla original."""

from dataclasses import replace

import pandas as pd

from tesina_backtest.backtest import run_backtest
from tesina_backtest.config import ROBUSTNESS_DIR, BacktestConfig
from tesina_backtest.experiments import MOMENTUM_WINDOWS
from tesina_backtest.reporting import save_results
from tesina_backtest.strategy import build_signals


def run(dataset: pd.DataFrame) -> None:

    base = BacktestConfig()
    summaries: list[pd.DataFrame] = []

    for window in MOMENTUM_WINDOWS:
        config = replace(base, momentum_window=window)
        result = run_backtest(build_signals(dataset, config), config)
        save_results(result, output_dir=ROBUSTNESS_DIR / f"momentum_{window}")

        summary = result.metrics.copy()
        summary.insert(0, "strategy", summary.index)
        summary.insert(0, "momentum_window", window)
        summaries.append(summary.reset_index(drop=True))

    combined = pd.concat(summaries, ignore_index=True)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ROBUSTNESS_DIR / "metrics_summary.csv", index=False)

    print(combined.round(4).to_string(index=False))
