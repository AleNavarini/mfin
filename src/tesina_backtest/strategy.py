from __future__ import annotations

import numpy as np
import pandas as pd

from tesina_backtest.config import BacktestConfig

FILTER_MODES = ("both", "panic", "euphoria", "none")


def calculate_cfgi_risk_regime(
    cfgi: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:

    if config.filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode debe ser uno de {FILTER_MODES}")

    uses_panic = config.filter_mode in ("both", "panic")
    uses_euphoria = config.filter_mode in ("both", "euphoria")

    prior_cfgi = cfgi.shift(1)

    history = prior_cfgi.expanding(min_periods=config.cfgi_warmup)

    thresholds = pd.DataFrame(
        {
            "p10": history.quantile(config.panic_enter_quantile),
            "p20": history.quantile(config.panic_exit_quantile),
            "p80": history.quantile(config.euphoria_exit_quantile),
            "p90": history.quantile(config.euphoria_enter_quantile),
        },
        index=cfgi.index,
    )

    regime = pd.Series(np.nan, index=cfgi.index, dtype=float, name="risk_regime")

    state = "normal"

    for date in cfgi.index:
        row = thresholds.loc[date]

        if row.isna().any():
            continue

        value = cfgi.loc[date]

        if state == "normal" and uses_panic and value <= row["p10"]:
            state = "panic"

        elif state == "normal" and uses_euphoria and value >= row["p90"]:
            state = "euphoria"

        elif (state == "panic" and value >= row["p20"]) or (
            state == "euphoria" and value <= row["p80"]
        ):
            state = "normal"

        regime.loc[date] = float(state == "normal")

    return regime, thresholds


def apply_no_trade_band(target: pd.Series, band: float) -> pd.Series:

    positions = pd.Series(np.nan, index=target.index, dtype=float, name="position")

    current = 0.0

    for date, desired in target.items():
        if pd.isna(desired):
            continue

        if desired == 0.0 or abs(desired - current) >= band:
            current = float(desired)

        positions.loc[date] = current

    return positions


def build_signals(dataset: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:

    frame = dataset.copy()

    # Retorno simple para el resultado de la cartera y logarítmico para la señal y la volatilidad.
    frame["btc_return"] = frame["close"].pct_change()

    frame["btc_log_return"] = np.log(frame["close"]).diff()

    frame["momentum"] = frame["btc_log_return"].rolling(config.momentum_window).sum()

    # Largo si el momentum es mayor o igual a cero, corto si es negativo. Vacío durante el calentamiento.
    frame["direction"] = np.where(frame["momentum"] >= 0, 1.0, -1.0)
    frame.loc[frame["momentum"].isna(), "direction"] = np.nan

    # EWMA de RiskMetrics: promedio exponencial de los retornos al cuadrado, con media supuesta cero.
    daily_variance = (
        frame["btc_log_return"]
        .pow(2)
        .ewm(
            alpha=1 - config.volatility_span_lambda,
            adjust=False,
            min_periods=config.volatility_min_periods,
        )
        .mean()
    )

    daily_volatility = np.sqrt(daily_variance)

    frame["annualized_volatility"] = daily_volatility * np.sqrt(config.annualization_days)

    frame["scaled_exposure"] = (config.target_volatility / frame["annualized_volatility"]).clip(
        upper=1.0
    )

    regime, thresholds = calculate_cfgi_risk_regime(frame["cfgi"], config)

    frame = frame.join(thresholds)
    frame["risk_regime"] = regime

    momentum_target = frame["direction"] * frame["scaled_exposure"]

    strategy_target = momentum_target * frame["risk_regime"]

    frame["momentum_target"] = momentum_target
    frame["strategy_target"] = strategy_target

    frame["momentum_position_signal"] = apply_no_trade_band(momentum_target, config.no_trade_band)
    frame["strategy_position_signal"] = apply_no_trade_band(strategy_target, config.no_trade_band)

    frame["momentum_position"] = frame["momentum_position_signal"].shift(1)
    frame["strategy_position"] = frame["strategy_position_signal"].shift(1)

    # Régimen vigente durante el día: la señal de t recién se aplica a la posición de t+1.
    frame["risk_regime_held"] = frame["risk_regime"].shift(1)

    return frame
