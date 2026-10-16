"""Análisis 3: descomposición del filtro en pánico y euforia, y grilla de percentiles."""

import pandas as pd

from tesina_backtest.backtest import BacktestResult, run_backtest
from tesina_backtest.config import ROBUSTNESS_DIR, BacktestConfig
from tesina_backtest.experiments import (
    MOMENTUM_WINDOWS,
    ORIGINAL_QUANTILES,
    QUANTILE_SETS,
    filter_config,
    format_quantiles,
)
from tesina_backtest.strategy import FILTER_MODES, build_signals


def _diagnostics(result: BacktestResult, config: BacktestConfig) -> dict[str, float]:

    frame = result.frame
    years = len(frame) / config.annualization_days

    out_of_market = frame["risk_regime_held"] == 0.0

    # Lo que habría ganado o perdido el momentum puro en los días que el filtro estuvo afuera.
    momentum_pnl_while_out = (
        frame.loc[out_of_market, "momentum_position"] * frame.loc[out_of_market, "btc_return"]
    )

    return {
        "observations": float(len(frame)),
        "days_in_refuge": float(out_of_market.sum()),
        "share_in_refuge": float(out_of_market.mean()),
        "avg_abs_exposure_strategy": float(frame["strategy_position"].abs().mean()),
        "avg_abs_exposure_momentum": float(frame["momentum_position"].abs().mean()),
        # Rotación sin contar la apertura del primer día. La rotación que cita la tesina,
        # que sí la cuenta, es la de analyses/costs.py.
        "turnover_strategy": float(frame["strategy_position"].diff().abs().sum() / years),
        "turnover_momentum": float(frame["momentum_position"].diff().abs().sum() / years),
        "momentum_pnl_forgone": float((1.0 + momentum_pnl_while_out).prod() - 1.0),
    }


def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:

    # Diferencias contra el momentum puro de la misma ventana. Un valor positivo favorece al filtro,
    # salvo en dMaxDD, donde el drawdown es negativo y un valor positivo indica una caída menor.
    filtered = summary.loc[summary["filter_mode"] != "none"].copy()
    filtered["label"] = filtered["filter_mode"] + "  " + filtered["quantiles"].str.slice(0, 9)
    filtered["dSharpe"] = filtered["strategy_sharpe"] - filtered["momentum_sharpe"]
    filtered["dRet"] = (
        filtered["strategy_annualized_return"] - filtered["momentum_annualized_return"]
    )
    filtered["dMaxDD"] = filtered["strategy_max_drawdown"] - filtered["momentum_max_drawdown"]
    filtered["dVol"] = (
        filtered["momentum_annualized_volatility"] - filtered["strategy_annualized_volatility"]
    )

    by_window = filtered.pivot(index="label", columns="momentum_window", values="dSharpe")
    by_window.columns = [f"dSh_{window}" for window in by_window.columns]

    grouped = filtered.groupby("label")
    table = by_window.assign(
        wins_Sharpe=grouped["dSharpe"].apply(lambda values: int((values > 0).sum())),
        avg_dSharpe=grouped["dSharpe"].mean(),
        avg_dRet=grouped["dRet"].mean(),
        avg_dMaxDD=grouped["dMaxDD"].mean(),
        avg_dVol=grouped["dVol"].mean(),
    )

    return table.sort_values("avg_dSharpe", ascending=False).reset_index()


def run(dataset: pd.DataFrame) -> None:

    base = BacktestConfig()
    rows: list[dict[str, float | str]] = []

    for window in MOMENTUM_WINDOWS:
        for mode in FILTER_MODES:
            # Sin filtro los percentiles no intervienen, así que alcanza con una corrida.
            quantile_sets = QUANTILE_SETS if mode != "none" else (ORIGINAL_QUANTILES,)

            for quantiles in quantile_sets:
                config = filter_config(base, window, mode, quantiles)
                result = run_backtest(build_signals(dataset, config), config)

                strategy_metrics = result.metrics.loc["strategy"].to_dict()
                momentum_metrics = result.metrics.loc["momentum"].to_dict()

                rows.append(
                    {
                        "momentum_window": window,
                        "filter_mode": mode,
                        "quantiles": "none" if mode == "none" else format_quantiles(quantiles),
                        **{f"strategy_{k}": v for k, v in strategy_metrics.items()},
                        **{f"momentum_{k}": v for k, v in momentum_metrics.items()},
                        **_diagnostics(result, config),
                    }
                )

    combined = pd.DataFrame(rows)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ROBUSTNESS_DIR / "cfgi_filter_window_summary.csv", index=False)

    combined.loc[combined["momentum_window"] == 90].drop(columns="momentum_window").to_csv(
        ROBUSTNESS_DIR / "cfgi_filter_summary.csv", index=False
    )
    _comparison_table(combined).to_csv(ROBUSTNESS_DIR / "cfgi_vs_momentum_table.csv", index=False)

    columns = [
        "momentum_window",
        "filter_mode",
        "quantiles",
        "momentum_annualized_return",
        "strategy_annualized_return",
        "strategy_annualized_volatility",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "share_in_refuge",
        "momentum_pnl_forgone",
        "turnover_strategy",
    ]

    print(combined[columns].round(4).to_string(index=False))
