# Permite usar anotaciones de tipos modernas en cualquier versión de Python.
from __future__ import annotations

# Path maneja rutas de archivos.
from pathlib import Path

# numpy para la raíz cuadrada al anualizar la volatilidad.
import numpy as np

# Estructura con los resultados de un backtest.
from tesina_backtest.backtest import BacktestResult


def save_results(result: BacktestResult, output_dir: Path) -> None:

    # Crea la carpeta de resultados si no existe.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tabla diaria completa con precios, señales y posiciones.
    result.frame.to_csv(output_dir / "backtest_daily.csv")

    # Retornos diarios de cada estrategia.
    result.returns.to_csv(output_dir / "daily_returns.csv")

    # Métricas agregadas (retorno, volatilidad, Sharpe, Sortino, drawdown).
    result.metrics.to_csv(output_dir / "metrics.csv")

    # Retorno de cada año calendario: producto de (1 + retorno diario) dentro del año, menos 1.
    annual_returns = (1.0 + result.returns).groupby(result.returns.index.year).prod() - 1.0
    annual_returns.index.name = "year"
    annual_returns.to_csv(output_dir / "annual_returns.csv")

    # Volatilidad de cada año calendario, anualizada con 365 días como en las métricas.
    annual_volatility = result.returns.groupby(result.returns.index.year).std() * np.sqrt(365)
    annual_volatility.index.name = "year"
    annual_volatility.to_csv(output_dir / "annual_volatility.csv")
