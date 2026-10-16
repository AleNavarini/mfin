"""Análisis 4: resultados netos de comisiones, slippage y funding."""

import pandas as pd

from tesina_backtest.backtest import BacktestResult, run_backtest
from tesina_backtest.config import COSTS_DIR, BacktestConfig
from tesina_backtest.experiments import (
    MOMENTUM_WINDOWS,
    ORIGINAL_QUANTILES,
    QUANTILE_SETS,
    annual_funding,
    annual_turnover,
    filter_config,
    format_quantiles,
)
from tesina_backtest.strategy import build_signals

FILTERS = {
    "regla original": ("both", ORIGINAL_QUANTILES),
    "panico 20/30": ("panic", (0.20, 0.30, 0.70, 0.80)),
}

GRID_MODES = ("both", "panic", "euphoria")


def _turnover_funding_row(
    result: BacktestResult, config: BacktestConfig, window: int, label: str
) -> dict[str, object]:

    frame = result.frame
    years = len(frame) / config.annualization_days
    row: dict[str, object] = {"momentum_window": window, "filtro": label}

    for name in ("strategy", "momentum"):
        position = frame[f"{name}_position"]
        row[f"{name}_turnover"] = annual_turnover(position, years)
        row[f"{name}_funding_annual"] = annual_funding(position, frame["funding_rate"], years)
        row[f"{name}_short_days"] = float((position < 0).mean())

    return row


def _grid_row(metrics: pd.DataFrame, window: int, mode: str, quantiles: tuple) -> dict[str, object]:

    return {
        "momentum_window": window,
        "filter_mode": mode,
        "quantiles": format_quantiles(quantiles),
        "strategy_return_gross": metrics.loc["strategy", "annualized_return"],
        "momentum_return_gross": metrics.loc["momentum", "annualized_return"],
        "strategy_volatility_net": metrics.loc["strategy_net", "annualized_volatility"],
        "momentum_volatility_net": metrics.loc["momentum_net", "annualized_volatility"],
        "strategy_sharpe_gross": metrics.loc["strategy", "sharpe"],
        "momentum_sharpe_gross": metrics.loc["momentum", "sharpe"],
        "strategy_sharpe_net": metrics.loc["strategy_net", "sharpe"],
        "momentum_sharpe_net": metrics.loc["momentum_net", "sharpe"],
        "strategy_return_net": metrics.loc["strategy_net", "annualized_return"],
        "momentum_return_net": metrics.loc["momentum_net", "annualized_return"],
        "strategy_max_drawdown_net": metrics.loc["strategy_net", "max_drawdown"],
        "momentum_max_drawdown_net": metrics.loc["momentum_net", "max_drawdown"],
    }


def run(dataset: pd.DataFrame) -> None:

    base = BacktestConfig()

    turnover_funding: list[dict[str, object]] = []
    grid: list[dict[str, object]] = []

    for window in MOMENTUM_WINDOWS:
        for label, (mode, quantiles) in FILTERS.items():
            config = filter_config(base, window, mode, quantiles)
            result = run_backtest(build_signals(dataset, config), config)
            turnover_funding.append(_turnover_funding_row(result, config, window, label))

        # Los costos de BacktestConfig ya están incluidos en las columnas *_net de las métricas.
        for mode in GRID_MODES:
            for quantiles in QUANTILE_SETS:
                config = filter_config(base, window, mode, quantiles)
                result = run_backtest(build_signals(dataset, config), config)
                grid.append(_grid_row(result.metrics, window, mode, quantiles))

    turnover_funding_frame = pd.DataFrame(turnover_funding)
    grid_frame = pd.DataFrame(grid)

    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    turnover_funding_frame.to_csv(COSTS_DIR / "turnover_funding.csv", index=False)
    grid_frame.to_csv(COSTS_DIR / "cost_grid_net.csv", index=False)

    pd.set_option("display.width", 200)

    print(turnover_funding_frame.round(4).to_string(index=False))
    print()

    grid_frame["gross_win"] = (
        grid_frame["strategy_sharpe_gross"] > grid_frame["momentum_sharpe_gross"]
    )
    grid_frame["net_win"] = grid_frame["strategy_sharpe_net"] > grid_frame["momentum_sharpe_net"]
    wins = grid_frame.groupby(["filter_mode", "quantiles"])[["gross_win", "net_win"]].sum()
    print(wins.to_string())
