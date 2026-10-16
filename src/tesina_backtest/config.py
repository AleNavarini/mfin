# dataclass genera automáticamente el constructor de una clase a partir de sus atributos.
from dataclasses import dataclass

# Path maneja rutas de archivos de forma independiente del sistema operativo.
from pathlib import Path


# frozen=True hace que la configuración no se pueda modificar una vez creada.
# Para cambiar un parámetro se crea una copia con dataclasses.replace.
@dataclass(frozen=True)
class BacktestConfig:
    # Inicio del período, alineado con el lanzamiento del perpetuo BTCUSDT el 10/09/2019.
    # Los días previos al lanzamiento usan funding cero.
    start_date: str = "2019-09-01"

    # Fecha de fin fija para que los resultados no cambien al descargar datos nuevos.
    end_date: str = "2026-09-19"

    # Días de retornos logarítmicos que suma la señal de momentum.
    momentum_window: int = 90

    # Volatilidad EWMA de RiskMetrics.
    # Lambda 0,94: cada día pesa 6% y el resto del peso queda en la estimación anterior.
    volatility_span_lambda: float = 0.94

    # Días mínimos de datos antes de que la volatilidad EWMA tenga valor.
    volatility_min_periods: int = 60

    # Volatilidad anual objetivo. La exposición se escala a objetivo / volatilidad, con tope en 1.
    target_volatility: float = 0.40

    # Observaciones previas del CFGI necesarias antes de calcular percentiles.
    cfgi_warmup: int = 365

    # Percentiles históricos del CFGI que abren y cierran cada régimen, con histéresis.
    # Se entra en pánico con el CFGI en o debajo del percentil 10.
    panic_enter_quantile: float = 0.10

    # Se sale del pánico cuando el CFGI vuelve al percentil 20 o más.
    panic_exit_quantile: float = 0.20

    # Se sale de la euforia cuando el CFGI baja al percentil 80 o menos.
    euphoria_exit_quantile: float = 0.80

    # Se entra en euforia con el CFGI en o encima del percentil 90.
    euphoria_enter_quantile: float = 0.90

    # "both", "panic", "euphoria" o "none".
    filter_mode: str = "both"

    # Cambio mínimo de exposición para rebalancear.
    no_trade_band: float = 0.10

    # Si es True, los retornos descuentan comisiones y slippage y suman o restan el funding.
    # Si es False, son retornos brutos.
    apply_costs: bool = True

    # Comisiones de un usuario regular de Binance que opera como tomador, sin descuento por BNB.
    # Contado: 10 puntos básicos (0,10%) por lado.
    spot_fee_bps: float = 10.0

    # Perpetuo: 5 puntos básicos (0,05%) por lado.
    perp_fee_bps: float = 5.0

    # Slippage por lado, por encima del medido en el libro de órdenes (ver run_slippage.py).
    slippage_bps: float = 2.0

    # Días por año para anualizar. BTC cotiza todos los días, por eso 365 y no 252.
    annualization_days: int = 365

    # Capital inicial de las curvas de capital.
    initial_cash: float = 100.0


# Carpeta raíz del proyecto: sube dos niveles desde src/tesina_backtest/config.py.
ROOT_DIR = Path(__file__).resolve().parents[2]


# Datos descargados tal como vienen de cada fuente.
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
# Dataset diario ya alineado que usan los backtests.
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
# Resultados de la grilla, con una subcarpeta para la corrida con costos y otra sin costos.
RESULTS_DIR = ROOT_DIR / "results"
