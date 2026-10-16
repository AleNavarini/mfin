from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt

from tesina_backtest.backtest import BacktestResult
from tesina_backtest.config import RESULTS_DIR


def save_results(result: BacktestResult, output_dir: Path = RESULTS_DIR) -> None:

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.frame.to_csv(output_dir / "backtest_daily.csv")

    result.returns.to_csv(output_dir / "daily_returns.csv")

    result.metrics.to_csv(output_dir / "metrics.csv")

    annual_returns = (1.0 + result.returns).groupby(result.returns.index.year).prod() - 1.0
    annual_returns.index.name = "year"
    annual_returns.to_csv(output_dir / "annual_returns.csv")

    # Volatilidad de cada año calendario, anualizada con 365 días como en las métricas.
    annual_volatility = result.returns.groupby(result.returns.index.year).std() * np.sqrt(365)
    annual_volatility.index.name = "year"
    annual_volatility.to_csv(output_dir / "annual_volatility.csv")

    _plot_equity(result.equity, figures_dir)
    _plot_drawdown(result.equity, figures_dir)
    _plot_positions(result.frame, figures_dir)
    _plot_cfgi(result.frame, figures_dir)
    _plot_interactive_equity(result.returns, figures_dir)
    _plot_interactive_drawdowns(result.returns, figures_dir)
    _plot_interactive_positions(result.frame, figures_dir)
    _plot_interactive_cfgi(result.frame, figures_dir)


def _plot_equity(equity: pd.DataFrame, figures_dir: Path) -> None:

    axis = equity.plot(figsize=(12, 6), title="Curvas de capital", linewidth=1.4)
    axis.set_ylabel("Capital inicial = 100")
    axis.set_xlabel("")

    axis.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(figures_dir / "equity_curves.png", dpi=180)
    plt.close()


def _plot_drawdown(equity: pd.DataFrame, figures_dir: Path) -> None:

    drawdown = equity / equity.cummax() - 1.0
    axis = drawdown.plot(figsize=(12, 5), title="Drawdowns", linewidth=1.2)
    axis.set_ylabel("Drawdown")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "drawdowns.png", dpi=180)
    plt.close()


def _plot_positions(frame: pd.DataFrame, figures_dir: Path) -> None:

    columns = ["strategy_position", "momentum_position"]
    axis = frame[columns].plot(figsize=(12, 5), title="Exposición a BTC", linewidth=1.0)

    axis.set_ylabel("Peso")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "positions.png", dpi=180)
    plt.close()


def _plot_cfgi(frame: pd.DataFrame, figures_dir: Path) -> None:

    columns = ["cfgi", "p10", "p20", "p80", "p90"]
    axis = frame[columns].plot(
        figsize=(12, 6),
        title="CFGI y umbrales históricos",
        linewidth=1.0,
    )

    axis.fill_between(
        frame.index,
        0,
        100,
        where=frame["risk_regime_held"].eq(0),
        alpha=0.12,
        color="red",
        label="Refugio",
    )

    axis.set_ylim(0, 100)
    axis.set_ylabel("Índice")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(figures_dir / "cfgi_regimes.png", dpi=180)
    plt.close()


def _returns_accessor(returns: pd.Series) -> vbt.returns.accessors.ReturnsSRAccessor:
    return vbt.returns.accessors.ReturnsSRAccessor(
        returns,
        freq="1D",
        year_freq="365D",
    )


def _write_html(figure: go.Figure, path: Path) -> None:
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def _plot_interactive_equity(returns: pd.DataFrame, figures_dir: Path) -> None:
    accessor = _returns_accessor(returns["strategy"])
    figure = accessor.plot_cumulative(
        benchmark_rets=returns["momentum"],
        start_value=100,
        title="Curvas de capital interactivas",
    )
    figure.data[0].name = "Estrategia con CFGI"
    figure.data[1].name = "Momentum"
    figure.update_layout(xaxis_title="", yaxis_title="Capital inicial = 100")
    _write_html(figure, figures_dir / "equity_curves_interactive.html")


def _plot_interactive_drawdowns(returns: pd.DataFrame, figures_dir: Path) -> None:
    for name, title in {
        "strategy": "Drawdowns de la estrategia con CFGI",
        "momentum": "Drawdowns de momentum",
    }.items():
        figure = _returns_accessor(returns[name]).drawdowns.plot(top_n=10, title=title)
        figure.update_layout(xaxis_title="", yaxis_title="Capital acumulado")
        _write_html(figure, figures_dir / f"drawdowns_{name}_interactive.html")


def _plot_interactive_positions(frame: pd.DataFrame, figures_dir: Path) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["strategy_position"],
            name="Estrategia con CFGI",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["momentum_position"],
            name="Momentum",
            mode="lines",
        )
    )
    figure.update_layout(
        title="Exposición interactiva a BTC",
        xaxis_title="",
        yaxis_title="Peso",
        hovermode="x unified",
    )
    _write_html(figure, figures_dir / "positions_interactive.html")


def _plot_interactive_cfgi(frame: pd.DataFrame, figures_dir: Path) -> None:
    figure = go.Figure()
    refuge = frame["risk_regime_held"].eq(0).astype(float).mul(100).replace(0, float("nan"))
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=refuge,
            name="Refugio",
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.12)",
            hoverinfo="skip",
        )
    )
    for column, name in {
        "cfgi": "CFGI",
        "p10": "p10",
        "p20": "p20",
        "p80": "p80",
        "p90": "p90",
    }.items():
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=name, mode="lines"))
    figure.update_layout(
        title="CFGI y umbrales históricos interactivos",
        xaxis_title="",
        yaxis_title="Índice",
        yaxis_range=[0, 100],
        hovermode="x unified",
    )
    _write_html(figure, figures_dir / "cfgi_regimes_interactive.html")
