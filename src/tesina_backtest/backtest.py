from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from tesina_backtest.config import BacktestConfig

# Cada estrategia y la columna con su posición diaria.
POSITION_COLUMNS = {"strategy": "strategy_position", "momentum": "momentum_position"}


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    returns: pd.DataFrame
    equity: pd.DataFrame
    metrics: pd.DataFrame


def annualized_tbill_return(dtb3: pd.Series, annualization_days: int) -> pd.Series:

    # DTB3 viene en porcentaje anual. Lo convertimos a una tasa diaria compuesta.
    annual_yield = dtb3.clip(lower=0.0) / 100.0

    return (1.0 + annual_yield) ** (1.0 / annualization_days) - 1.0


def net_of_costs(
    gross: pd.Series,
    position: pd.Series,
    funding_rate: pd.Series,
    spot_cost: float,
    perp_cost: float,
) -> pd.Series:

    # Los largos se operan al contado y los cortos en el perpetuo, cada uno con su costo por lado.
    # El primer día se abre la posición desde cero.
    long_leg = position.clip(lower=0.0)
    short_leg = position.clip(upper=0.0)

    spot_traded = long_leg.diff().fillna(long_leg).abs()
    perp_traded = short_leg.diff().fillna(short_leg).abs()

    # Los cortos cobran el funding cuando la tasa es positiva y lo pagan cuando es negativa.
    funding = -short_leg * funding_rate

    return gross - spot_cost * spot_traded - perp_cost * perp_traded + funding


def compute_metrics(
    returns: pd.DataFrame, risk_free: pd.Series, config: BacktestConfig
) -> pd.DataFrame:

    rows: dict[str, dict[str, float]] = {}

    for name in returns.columns:
        accessor = vbt.returns.accessors.ReturnsSRAccessor(
            returns[name],
            freq="1D",
            year_freq=f"{config.annualization_days}D",
        )

        # Sharpe y Sortino se calculan sobre el exceso de retorno respecto del activo de refugio.
        excess = vbt.returns.accessors.ReturnsSRAccessor(
            returns[name] - risk_free,
            freq="1D",
            year_freq=f"{config.annualization_days}D",
        )

        rows[name] = {
            "total_return": accessor.total(),
            "annualized_return": accessor.annualized(),
            "annualized_volatility": accessor.annualized_volatility(),
            "sharpe": excess.sharpe_ratio(),
            "sortino": excess.sortino_ratio(),
            "max_drawdown": accessor.max_drawdown(),
        }

    return pd.DataFrame(rows).T


def run_backtest(signal_frame: pd.DataFrame, config: BacktestConfig) -> BacktestResult:

    period = (signal_frame.index >= pd.Timestamp(config.start_date)) & (
        signal_frame.index <= pd.Timestamp(config.end_date)
    )
    frame = signal_frame.loc[period].copy()

    required = ["btc_return", "dtb3", "funding_rate", "momentum_position", "strategy_position"]

    frame = frame.dropna(subset=required)
    if frame.empty:
        raise ValueError("No existen observaciones operables para el período solicitado")

    frame["refuge_return"] = annualized_tbill_return(frame["dtb3"], config.annualization_days)

    returns = pd.DataFrame(index=frame.index)

    # La fracción no expuesta a BTC queda en el activo de refugio.
    for name, position_column in POSITION_COLUMNS.items():
        position = frame[position_column]

        refuge_weight = 1.0 - position.abs()

        returns[name] = position * frame["btc_return"] + refuge_weight * frame["refuge_return"]

    spot_cost = (config.spot_fee_bps + config.slippage_bps) / 10_000
    perp_cost = (config.perp_fee_bps + config.slippage_bps) / 10_000

    for name, position_column in POSITION_COLUMNS.items():
        returns[f"{name}_net"] = net_of_costs(
            returns[name], frame[position_column], frame["funding_rate"], spot_cost, perp_cost
        )

    equity = config.initial_cash * (1.0 + returns).cumprod()

    metrics = compute_metrics(returns, frame["refuge_return"], config)

    return BacktestResult(frame, returns, equity, metrics)
