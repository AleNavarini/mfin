"""Guardado de los resultados de un backtest en CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tesina_backtest.backtest import BacktestResult


def save_results(result: BacktestResult, output_dir: Path) -> None:
    """Guarda los resultados de un backtest en `output_dir`.

    Escribe la tabla diaria completa (`backtest_daily.csv`), los retornos diarios
    (`daily_returns.csv`), las métricas agregadas (`metrics.csv`) y el retorno y la
    volatilidad anualizada de cada año calendario (`annual_returns.csv` y
    `annual_volatility.csv`).

    Args:
        result: Resultado de `run_backtest`.
        output_dir: Carpeta de salida. Se crea si no existe.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    result.frame.to_csv(output_dir / "backtest_daily.csv")
    result.returns.to_csv(output_dir / "daily_returns.csv")
    result.metrics.to_csv(output_dir / "metrics.csv")

    annual_returns = (1.0 + result.returns).groupby(result.returns.index.year).prod() - 1.0
    annual_returns.index.name = "year"
    annual_returns.to_csv(output_dir / "annual_returns.csv")

    annual_volatility = result.returns.groupby(result.returns.index.year).std() * np.sqrt(365)
    annual_volatility.index.name = "year"
    annual_volatility.to_csv(output_dir / "annual_volatility.csv")
