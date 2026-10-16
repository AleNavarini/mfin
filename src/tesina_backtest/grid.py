"""Grilla única de configuraciones: ventana de momentum, modo del filtro y percentiles."""

# Permite usar anotaciones de tipos modernas en cualquier versión de Python.
from __future__ import annotations

# Iterator describe el tipo de una función que devuelve valores de a uno (con yield).
from collections.abc import Iterator

# replace crea una copia de la configuración con algunos campos cambiados.
from dataclasses import replace

# pandas para armar las tablas de resultados.
import pandas as pd

# Función que corre el backtest y estructura con sus resultados.
from tesina_backtest.backtest import BacktestResult, run_backtest

# Carpeta de resultados y parámetros por defecto.
from tesina_backtest.config import RESULTS_DIR, BacktestConfig

# Función que guarda los CSV de una corrida.
from tesina_backtest.reporting import save_results

# Modos del filtro y función que calcula las señales.
from tesina_backtest.strategy import FILTER_MODES, build_signals

# Ventanas de momentum (en días) que recorre la grilla.
MOMENTUM_WINDOWS = (30, 60, 90, 180, 365)

# Cada juego es (entrada pánico, salida pánico, salida euforia, entrada euforia).
QUANTILE_SETS = (
    # El más extremo: el filtro actúa pocas veces.
    (0.05, 0.10, 0.90, 0.95),
    # El de la regla original.
    (0.10, 0.20, 0.80, 0.90),
    (0.15, 0.25, 0.75, 0.85),
    # El más amplio: el filtro actúa más seguido.
    (0.20, 0.30, 0.70, 0.80),
)


def grid_configs(base: BacktestConfig) -> Iterator[BacktestConfig]:

    # Recorre las cinco ventanas de momentum.
    for window in MOMENTUM_WINDOWS:
        # Recorre los cuatro modos: ambos, pánico, euforia y ninguno.
        for mode in FILTER_MODES:
            # Sin filtro los percentiles no intervienen, así que alcanza con una corrida.
            quantile_sets = QUANTILE_SETS if mode != "none" else QUANTILE_SETS[1:2]

            # Recorre los juegos de percentiles.
            for panic_enter, panic_exit, euphoria_exit, euphoria_enter in quantile_sets:
                # Devuelve una copia de la configuración base con ventana, modo y percentiles nuevos.
                yield replace(
                    base,
                    momentum_window=window,
                    filter_mode=mode,
                    panic_enter_quantile=panic_enter,
                    panic_exit_quantile=panic_exit,
                    euphoria_exit_quantile=euphoria_exit,
                    euphoria_enter_quantile=euphoria_enter,
                )


def _quantiles(config: BacktestConfig) -> list[float]:
    # Los cuatro percentiles de la configuración, en el mismo orden que QUANTILE_SETS.
    return [
        config.panic_enter_quantile,
        config.panic_exit_quantile,
        config.euphoria_exit_quantile,
        config.euphoria_enter_quantile,
    ]


def _run_name(config: BacktestConfig) -> str:
    # Nombre de la carpeta de cada corrida, por ejemplo "90_panic_0.20-0.30-0.70-0.80" o "90_none".
    if config.filter_mode == "none":
        return f"{config.momentum_window}_none"
    quantiles = "-".join(f"{q:.2f}" for q in _quantiles(config))
    return f"{config.momentum_window}_{config.filter_mode}_{quantiles}"


def _summary_row(result: BacktestResult, config: BacktestConfig) -> dict[str, object]:

    # Tabla diaria del backtest.
    frame = result.frame
    # Cantidad de años del período: días observados sobre 365.
    years = len(frame) / config.annualization_days

    # Identificación de la corrida. Sin filtro los percentiles no aplican.
    row: dict[str, object] = {
        "momentum_window": config.momentum_window,
        "filter_mode": config.filter_mode,
        "quantiles": "none"
        if config.filter_mode == "none"
        else "/".join(f"{q:.2f}" for q in _quantiles(config)),
    }

    # Las mismas medidas para la estrategia con filtro y para el momentum puro.
    for name in ("strategy", "momentum"):
        # Métricas agregadas con prefijo, por ejemplo strategy_sharpe o momentum_max_drawdown.
        for metric, value in result.metrics.loc[name].items():
            row[f"{name}_{metric}"] = value

        # Posición diaria de esa estrategia.
        position = frame[f"{name}_position"]
        # Exposición absoluta promedio a BTC.
        row[f"{name}_avg_abs_exposure"] = float(position.abs().mean())
        # Rotación anual: cada cambio de posición se opera por su valor absoluto.
        # El primer día se abre la posición desde cero.
        row[f"{name}_turnover"] = float(position.diff().fillna(position.abs()).abs().sum() / years)
        # Funding anual que cobra (positivo) o paga (negativo) la pata corta en el perpetuo.
        # Se informa siempre, aunque solo entra en los retornos si apply_costs es True.
        row[f"{name}_funding_annual"] = float(
            (-position.clip(upper=0.0) * frame["funding_rate"]).sum() / years
        )
        # Fracción de días en posición corta.
        row[f"{name}_short_days"] = float((position < 0).mean())

    # True en los días en que la estrategia estuvo en refugio por el filtro.
    out_of_market = frame["risk_regime_held"] == 0.0

    # Lo que habría ganado o perdido el momentum puro (bruto) en los días que el filtro estuvo afuera.
    momentum_pnl_while_out = (
        frame.loc[out_of_market, "momentum_position"] * frame.loc[out_of_market, "btc_return"]
    )

    # Días totales, días en refugio, fracción en refugio y retorno resignado.
    row["observations"] = float(len(frame))
    row["days_in_refuge"] = float(out_of_market.sum())
    row["share_in_refuge"] = float(out_of_market.mean())
    row["momentum_pnl_forgone"] = float((1.0 + momentum_pnl_while_out).prod() - 1.0)

    return row


def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:

    # Diferencias contra el momentum puro de la misma ventana. Un valor positivo favorece al filtro,
    # salvo en dMaxDD, donde el drawdown es negativo y un valor positivo indica una caída menor.
    # Se queda solo con las corridas que tienen filtro.
    filtered = summary.loc[summary["filter_mode"] != "none"].copy()
    # Etiqueta de cada combinación: modo y los dos primeros percentiles, por ejemplo "panic  0.20/0.30".
    filtered["label"] = filtered["filter_mode"] + "  " + filtered["quantiles"].str.slice(0, 9)
    # Diferencia de Sharpe: filtro menos momentum puro.
    filtered["dSharpe"] = filtered["strategy_sharpe"] - filtered["momentum_sharpe"]
    # Diferencia de retorno anualizado: filtro menos momentum puro.
    filtered["dRet"] = (
        filtered["strategy_annualized_return"] - filtered["momentum_annualized_return"]
    )
    # Diferencia de máximo drawdown: filtro menos momentum puro.
    filtered["dMaxDD"] = filtered["strategy_max_drawdown"] - filtered["momentum_max_drawdown"]
    # Diferencia de volatilidad al revés (momentum menos filtro), para que positivo sea mejor.
    filtered["dVol"] = (
        filtered["momentum_annualized_volatility"] - filtered["strategy_annualized_volatility"]
    )

    # Tabla con una fila por combinación y una columna con la diferencia de Sharpe por ventana.
    by_window = filtered.pivot(index="label", columns="momentum_window", values="dSharpe")
    # Renombra las columnas a dSh_30, dSh_60, etc.
    by_window.columns = [f"dSh_{window}" for window in by_window.columns]

    # Agrupa las cinco ventanas de cada combinación.
    grouped = filtered.groupby("label")
    table = by_window.assign(
        # En cuántas de las cinco ventanas el filtro supera al momentum puro en Sharpe.
        wins_Sharpe=grouped["dSharpe"].apply(lambda values: int((values > 0).sum())),
        # Promedio de cada diferencia entre las cinco ventanas.
        avg_dSharpe=grouped["dSharpe"].mean(),
        avg_dRet=grouped["dRet"].mean(),
        avg_dMaxDD=grouped["dMaxDD"].mean(),
        avg_dVol=grouped["dVol"].mean(),
    )

    # Ordena de la combinación con mayor mejora promedio de Sharpe a la de menor.
    return table.sort_values("avg_dSharpe", ascending=False).reset_index()


def run_grid(dataset: pd.DataFrame, base: BacktestConfig) -> None:

    # Carpeta de salida según si la corrida incluye costos o no.
    output_dir = RESULTS_DIR / ("con_costos" if base.apply_costs else "sin_costos")
    # Una fila de resultados por configuración.
    rows: list[dict[str, object]] = []

    # Corre cada configuración de la grilla una sola vez.
    for config in grid_configs(base):
        # Calcula señales y corre el backtest.
        result = run_backtest(build_signals(dataset, config), config)
        # Guarda el detalle diario, las métricas y los retornos por año en runs/<nombre>/.
        save_results(result, output_dir / "runs" / _run_name(config))
        # Agrega la fila de resumen de esta configuración.
        rows.append(_summary_row(result, config))

    # Junta todas las filas en una tabla y la guarda.
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "grilla.csv", index=False)

    # Guarda la tabla de diferencias contra el momentum puro.
    comparison = _comparison_table(summary)
    comparison.to_csv(output_dir / "comparacion.csv", index=False)

    # Muestra la comparación en la terminal, con una salida ancha para que no se corte.
    pd.set_option("display.width", 200)
    print(comparison.round(4).to_string(index=False))
