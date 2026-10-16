# Permite usar anotaciones de tipos modernas en cualquier versión de Python.
from __future__ import annotations

# numpy hace cálculos numéricos vectorizados (raíz cuadrada, logaritmo, condiciones).
import numpy as np

# pandas maneja las series de tiempo diarias.
import pandas as pd

# Parámetros del backtest (ventana, volatilidad objetivo, percentiles, etc.).
from tesina_backtest.config import BacktestConfig

# Modos válidos del filtro: ambos regímenes, solo pánico, solo euforia o ninguno.
FILTER_MODES = ("both", "panic", "euphoria", "none")


def calculate_cfgi_risk_regime(
    cfgi: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:

    # Rechaza un modo de filtro que no exista.
    if config.filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode debe ser uno de {FILTER_MODES}")

    # True si el modo elegido activa el régimen de pánico.
    uses_panic = config.filter_mode in ("both", "panic")
    # True si el modo elegido activa el régimen de euforia.
    uses_euphoria = config.filter_mode in ("both", "euphoria")

    # Corre el CFGI un día hacia atrás: los percentiles de hoy solo usan datos hasta ayer.
    # Así se evita usar información del futuro (look-ahead bias).
    prior_cfgi = cfgi.shift(1)

    # Ventana expansiva: cada día toma toda la historia previa. No da valor hasta tener 365 días.
    history = prior_cfgi.expanding(min_periods=config.cfgi_warmup)

    # Calcula para cada día los cuatro umbrales históricos del CFGI.
    thresholds = pd.DataFrame(
        {
            # Umbral para entrar en pánico.
            "panic_enter": history.quantile(config.panic_enter_quantile),
            # Umbral para salir del pánico.
            "panic_exit": history.quantile(config.panic_exit_quantile),
            # Umbral para salir de la euforia.
            "euphoria_exit": history.quantile(config.euphoria_exit_quantile),
            # Umbral para entrar en euforia.
            "euphoria_enter": history.quantile(config.euphoria_enter_quantile),
        },
        index=cfgi.index,
    )

    # Serie del régimen: 1 = normal (se opera), 0 = pánico o euforia (refugio). Vacía al inicio.
    regime = pd.Series(np.nan, index=cfgi.index, dtype=float, name="risk_regime")

    # Estado inicial antes de recorrer los días.
    state = "normal"

    # Recorre día por día porque el estado de hoy depende del de ayer (histéresis).
    for date in cfgi.index:
        # Umbrales de ese día.
        row = thresholds.loc[date]

        # Durante el calentamiento los umbrales están vacíos. Se saltea el día y el régimen queda vacío.
        if row.isna().any():
            continue

        # Valor del CFGI de ese día.
        value = cfgi.loc[date]

        # Desde normal, entra en pánico si el CFGI cae al umbral de entrada al pánico o menos.
        if state == "normal" and uses_panic and value <= row["panic_enter"]:
            state = "panic"

        # Desde normal, entra en euforia si el CFGI sube al umbral de entrada a la euforia o más.
        elif state == "normal" and uses_euphoria and value >= row["euphoria_enter"]:
            state = "euphoria"

        # Vuelve a normal si el CFGI cruza el umbral de salida del pánico o de la euforia.
        # Como los umbrales de salida son distintos de los de entrada, no hay idas y vueltas diarias.
        elif (state == "panic" and value >= row["panic_exit"]) or (
            state == "euphoria" and value <= row["euphoria_exit"]
        ):
            state = "normal"

        # Guarda 1.0 si el estado es normal y 0.0 si es pánico o euforia.
        regime.loc[date] = float(state == "normal")

    # Devuelve el régimen diario y los umbrales.
    return regime, thresholds


def apply_no_trade_band(target: pd.Series, band: float) -> pd.Series:

    # Serie de posiciones efectivas, vacía al inicio.
    positions = pd.Series(np.nan, index=target.index, dtype=float, name="position")

    # Posición actual. Arranca sin exposición.
    current = 0.0

    # Recorre día por día porque cada decisión depende de la posición anterior.
    for date, desired in target.items():
        # Durante el calentamiento no hay objetivo. Se deja el día vacío.
        if pd.isna(desired):
            continue

        # Rebalancea si el objetivo es cerrar todo (ir a refugio) o si el cambio es de al menos la banda.
        # Si no, mantiene la posición anterior y se ahorra el costo de operar cambios chicos.
        if desired == 0.0 or abs(desired - current) >= band:
            current = float(desired)

        # Registra la posición vigente de ese día.
        positions.loc[date] = current

    return positions


def build_signals(dataset: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:

    # Copia el dataset para no modificar el original.
    frame = dataset.copy()

    # Retorno simple para el resultado de la cartera y logarítmico para la señal y la volatilidad.
    # Retorno simple diario: cierre de hoy / cierre de ayer - 1.
    frame["btc_return"] = frame["close"].pct_change()

    # Retorno logarítmico diario: log(cierre de hoy) - log(cierre de ayer).
    frame["btc_log_return"] = np.log(frame["close"]).diff()

    # Momentum: suma de los retornos logarítmicos de los últimos N días (90 por defecto).
    frame["momentum"] = frame["btc_log_return"].rolling(config.momentum_window).sum()

    # Largo si el momentum es mayor o igual a cero, corto si es negativo. Vacío durante el calentamiento.
    frame["direction"] = np.where(frame["momentum"] >= 0, 1.0, -1.0)
    # np.where pone -1 donde el momentum está vacío. Se corrige dejando esos días vacíos.
    frame.loc[frame["momentum"].isna(), "direction"] = np.nan

    # EWMA de RiskMetrics: promedio exponencial de los retornos al cuadrado, con media supuesta cero.
    daily_variance = (
        # Retornos logarítmicos diarios.
        frame["btc_log_return"]
        # Al cuadrado: con media cero, eso es la varianza de cada día.
        .pow(2)
        # Promedio exponencial: alpha = 1 - lambda = 0,06 es el peso del día nuevo.
        # adjust=False usa la fórmula recursiva de RiskMetrics.
        # min_periods exige 60 días de datos antes de dar un valor.
        .ewm(
            alpha=1 - config.volatility_span_lambda,
            adjust=False,
            min_periods=config.volatility_min_periods,
        )
        .mean()
    )

    # Volatilidad diaria: raíz cuadrada de la varianza.
    daily_volatility = np.sqrt(daily_variance)

    # Volatilidad anual: la diaria por la raíz de 365.
    frame["annualized_volatility"] = daily_volatility * np.sqrt(config.annualization_days)

    # Exposición escalada: volatilidad objetivo (40%) / volatilidad estimada, con tope en 1 (sin apalancamiento).
    # Si BTC está más volátil, se expone menos capital.
    frame["scaled_exposure"] = (config.target_volatility / frame["annualized_volatility"]).clip(
        upper=1.0
    )

    # Calcula el régimen del CFGI (1 = opera, 0 = refugio) y los umbrales diarios.
    regime, thresholds = calculate_cfgi_risk_regime(frame["cfgi"], config)

    # Agrega las columnas de los cuatro umbrales a la tabla.
    frame = frame.join(thresholds)
    # Agrega el régimen a la tabla.
    frame["risk_regime"] = regime

    # Objetivo del momentum puro: dirección (+1 o -1) por exposición escalada.
    momentum_target = frame["direction"] * frame["scaled_exposure"]

    # Objetivo de la estrategia con CFGI: el mismo, pero en cero cuando el régimen manda a refugio.
    strategy_target = momentum_target * frame["risk_regime"]

    # Guarda ambos objetivos en la tabla.
    frame["momentum_target"] = momentum_target
    frame["strategy_target"] = strategy_target

    # Aplica la banda de no operación a cada objetivo para obtener la posición decidida al cierre de cada día.
    frame["momentum_position_signal"] = apply_no_trade_band(momentum_target, config.no_trade_band)
    frame["strategy_position_signal"] = apply_no_trade_band(strategy_target, config.no_trade_band)

    # La posición decidida al cierre de t se mantiene durante t+1. Por eso se corre un día.
    # Así el retorno de cada día usa una posición tomada con información del día anterior.
    frame["momentum_position"] = frame["momentum_position_signal"].shift(1)
    frame["strategy_position"] = frame["strategy_position_signal"].shift(1)

    # Régimen vigente durante el día: la señal de t recién se aplica a la posición de t+1.
    frame["risk_regime_held"] = frame["risk_regime"].shift(1)

    return frame
