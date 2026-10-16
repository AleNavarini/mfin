# Permite usar anotaciones de tipos modernas en cualquier versión de Python.
from __future__ import annotations

# dataclass genera el constructor de la clase de resultados.
from dataclasses import dataclass

# pandas maneja las series de retornos.
import pandas as pd

# vectorbt calcula las métricas de desempeño (Sharpe, Sortino, drawdown, etc.).
import vectorbt as vbt

# Parámetros del backtest.
from tesina_backtest.config import BacktestConfig

# Cada estrategia y la columna con su posición diaria.
POSITION_COLUMNS = {"strategy": "strategy_position", "momentum": "momentum_position"}


# Agrupa todo lo que produce un backtest.
@dataclass
class BacktestResult:
    # Tabla diaria con precios, señales y posiciones.
    frame: pd.DataFrame
    # Retornos diarios de cada estrategia, netos o brutos según apply_costs.
    returns: pd.DataFrame
    # Curvas de capital (arrancan en 100).
    equity: pd.DataFrame
    # Métricas agregadas de cada estrategia.
    metrics: pd.DataFrame


def annualized_tbill_return(dtb3: pd.Series, annualization_days: int) -> pd.Series:

    # DTB3 viene en porcentaje anual. Lo convertimos a una tasa diaria compuesta.
    # Recorta tasas negativas a cero y divide por 100 (por ejemplo 5,0 pasa a 0,05).
    annual_yield = dtb3.clip(lower=0.0) / 100.0

    # Tasa diaria equivalente: (1 + tasa anual) elevado a 1/365, menos 1.
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
    # Parte larga de la posición: los valores positivos, el resto en cero.
    long_leg = position.clip(lower=0.0)
    # Parte corta de la posición: los valores negativos, el resto en cero.
    short_leg = position.clip(upper=0.0)

    # Monto operado cada día al contado: cambio absoluto de la pata larga.
    # fillna cubre el primer día, cuando se abre toda la posición.
    spot_traded = long_leg.diff().fillna(long_leg).abs()
    # Monto operado cada día en el perpetuo: cambio absoluto de la pata corta.
    perp_traded = short_leg.diff().fillna(short_leg).abs()

    # Los cortos cobran el funding cuando la tasa es positiva y lo pagan cuando es negativa.
    # short_leg es negativo, así que -short_leg es el tamaño del corto.
    funding = -short_leg * funding_rate

    # Retorno neto: bruto menos costo de operar en cada mercado, más el funding.
    return gross - spot_cost * spot_traded - perp_cost * perp_traded + funding


def compute_metrics(
    returns: pd.DataFrame, risk_free: pd.Series, config: BacktestConfig
) -> pd.DataFrame:

    # Una fila de métricas por estrategia.
    rows: dict[str, dict[str, float]] = {}

    # Recorre cada columna de retornos (strategy y momentum).
    for name in returns.columns:
        # Accesor de vectorbt sobre los retornos diarios, con año de 365 días.
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
            # Retorno acumulado de todo el período.
            "total_return": accessor.total(),
            # Retorno anual compuesto equivalente.
            "annualized_return": accessor.annualized(),
            # Desvío estándar diario por la raíz de 365.
            "annualized_volatility": accessor.annualized_volatility(),
            # Exceso de retorno medio sobre su desvío, anualizado.
            "sharpe": excess.sharpe_ratio(),
            # Igual que Sharpe pero solo penaliza la volatilidad a la baja.
            "sortino": excess.sortino_ratio(),
            # Peor caída desde un máximo previo de la curva de capital.
            "max_drawdown": accessor.max_drawdown(),
        }

    # Convierte el diccionario en tabla con una fila por estrategia.
    return pd.DataFrame(rows).T


def run_backtest(signal_frame: pd.DataFrame, config: BacktestConfig) -> BacktestResult:

    # Marca los días que caen dentro del período de evaluación.
    period = (signal_frame.index >= pd.Timestamp(config.start_date)) & (
        signal_frame.index <= pd.Timestamp(config.end_date)
    )
    # Se queda solo con esos días.
    frame = signal_frame.loc[period].copy()

    # Columnas que deben tener valor para poder calcular el retorno de un día.
    required = ["btc_return", "dtb3", "funding_rate", "momentum_position", "strategy_position"]

    # Descarta los días con alguna de esas columnas vacía (por ejemplo, el calentamiento).
    frame = frame.dropna(subset=required)
    # Si no queda ningún día, no se puede hacer el backtest.
    if frame.empty:
        raise ValueError("No existen observaciones operables para el período solicitado")

    # Retorno diario del activo de refugio, a partir de la tasa DTB3.
    frame["refuge_return"] = annualized_tbill_return(frame["dtb3"], config.annualization_days)

    # Tabla vacía de retornos diarios, con las mismas fechas.
    returns = pd.DataFrame(index=frame.index)

    # Costo por lado al contado: comisión más slippage, pasado de puntos básicos a fracción.
    spot_cost = (config.spot_fee_bps + config.slippage_bps) / 10_000
    # Costo por lado en el perpetuo: comisión más slippage.
    perp_cost = (config.perp_fee_bps + config.slippage_bps) / 10_000

    # La fracción no expuesta a BTC queda en el activo de refugio.
    for name, position_column in POSITION_COLUMNS.items():
        # Posición del día (entre -1 y 1).
        position = frame[position_column]

        # Peso en refugio: lo que no está expuesto a BTC, sea largo o corto.
        refuge_weight = 1.0 - position.abs()

        # Retorno bruto: posición por retorno de BTC más peso en refugio por retorno del refugio.
        returns[name] = position * frame["btc_return"] + refuge_weight * frame["refuge_return"]

        # Con costos activados, descuenta comisiones y slippage y suma o resta el funding.
        if config.apply_costs:
            returns[name] = net_of_costs(
                returns[name], position, frame["funding_rate"], spot_cost, perp_cost
            )

    # Curva de capital: 100 por el producto acumulado de (1 + retorno diario).
    equity = config.initial_cash * (1.0 + returns).cumprod()

    # Métricas de cada columna, usando el refugio como tasa libre de riesgo.
    metrics = compute_metrics(returns, frame["refuge_return"], config)

    return BacktestResult(frame, returns, equity, metrics)
