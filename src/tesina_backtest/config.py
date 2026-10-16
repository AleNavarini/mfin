from dataclasses import dataclass
from pathlib import Path


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
    volatility_span_lambda: float = 0.94

    volatility_min_periods: int = 60

    # Volatilidad anual objetivo. La exposición se escala a objetivo / volatilidad, con tope en 1.
    target_volatility: float = 0.40

    # Observaciones previas del CFGI necesarias antes de calcular percentiles.
    cfgi_warmup: int = 365

    # Percentiles históricos del CFGI que abren y cierran cada régimen, con histéresis.
    panic_enter_quantile: float = 0.10

    panic_exit_quantile: float = 0.20

    euphoria_exit_quantile: float = 0.80

    euphoria_enter_quantile: float = 0.90

    # "both", "panic", "euphoria" o "none".
    filter_mode: str = "both"

    # Cambio mínimo de exposición para rebalancear.
    no_trade_band: float = 0.10

    # Comisiones de un usuario regular de Binance que opera como tomador, sin descuento por BNB.
    spot_fee_bps: float = 10.0

    perp_fee_bps: float = 5.0

    # Slippage por lado, por encima del medido en el libro de órdenes (ver run_slippage.py).
    slippage_bps: float = 2.0

    annualization_days: int = 365

    initial_cash: float = 100.0


ROOT_DIR = Path(__file__).resolve().parents[2]


DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ROBUSTNESS_DIR = RESULTS_DIR / "robustez"
COSTS_DIR = RESULTS_DIR / "costos"
SLIPPAGE_DIR = RESULTS_DIR / "slippage"
