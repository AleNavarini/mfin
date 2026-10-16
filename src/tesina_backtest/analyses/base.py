"""Análisis 1: backtest base de la regla original contra el momentum puro."""

import pandas as pd

from tesina_backtest.backtest import run_backtest
from tesina_backtest.config import RESULTS_DIR, BacktestConfig
from tesina_backtest.reporting import save_results
from tesina_backtest.strategy import build_signals


def run(dataset: pd.DataFrame) -> None:

    config = BacktestConfig()

    result = run_backtest(build_signals(dataset, config), config)

    save_results(result, output_dir=RESULTS_DIR)

    print(result.metrics.round(4).to_string())
