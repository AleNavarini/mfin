"""Parámetros compartidos por los análisis de robustez y costos."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tesina_backtest.config import BacktestConfig

MOMENTUM_WINDOWS = (30, 60, 90, 180, 365)

# Cada juego es (entrada pánico, salida pánico, salida euforia, entrada euforia).
QUANTILE_SETS = (
    (0.05, 0.10, 0.90, 0.95),
    (0.10, 0.20, 0.80, 0.90),
    (0.15, 0.25, 0.75, 0.85),
    (0.20, 0.30, 0.70, 0.80),
)

ORIGINAL_QUANTILES = QUANTILE_SETS[1]

Quantiles = tuple[float, float, float, float]


def filter_config(
    base: BacktestConfig, window: int, mode: str, quantiles: Quantiles
) -> BacktestConfig:

    panic_enter, panic_exit, euphoria_exit, euphoria_enter = quantiles

    return replace(
        base,
        momentum_window=window,
        filter_mode=mode,
        panic_enter_quantile=panic_enter,
        panic_exit_quantile=panic_exit,
        euphoria_exit_quantile=euphoria_exit,
        euphoria_enter_quantile=euphoria_enter,
    )


def format_quantiles(quantiles: Quantiles) -> str:
    return "/".join(f"{q:.2f}" for q in quantiles)


def annual_turnover(position: pd.Series, years: float) -> float:

    # Cada cambio de posición se opera por su valor absoluto. El primer día se abre desde cero.
    traded = position.diff().fillna(position.abs()).abs()

    return float(traded.sum() / years)


def annual_funding(position: pd.Series, funding_rate: pd.Series, years: float) -> float:

    # Solo la pata corta, que se opera en el perpetuo, cobra o paga funding.
    return float((-position.clip(upper=0.0) * funding_rate).sum() / years)
