# Permite usar anotaciones de tipos modernas en cualquier versión de Python.
from __future__ import annotations

# Path maneja rutas de archivos.
from pathlib import Path

# matplotlib dibuja los gráficos estáticos en PNG.
import matplotlib.pyplot as plt

# numpy para la raíz cuadrada al anualizar la volatilidad.
import numpy as np

# pandas maneja las tablas de resultados.
import pandas as pd

# plotly dibuja los gráficos interactivos en HTML.
import plotly.graph_objects as go

# vectorbt arma algunos gráficos interactivos a partir de los retornos.
import vectorbt as vbt

# Estructura con los resultados de un backtest.
from tesina_backtest.backtest import BacktestResult


def save_results(result: BacktestResult, output_dir: Path, figures: bool) -> None:

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

    # Los gráficos solo se generan si se piden, porque son lentos.
    if not figures:
        return

    # Crea la carpeta de gráficos.
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Gráficos estáticos en PNG.
    _plot_equity(result.equity, figures_dir)
    _plot_drawdown(result.equity, figures_dir)
    _plot_positions(result.frame, figures_dir)
    _plot_cfgi(result.frame, figures_dir)
    # Gráficos interactivos en HTML.
    _plot_interactive_equity(result.returns, figures_dir)
    _plot_interactive_drawdowns(result.returns, figures_dir)
    _plot_interactive_positions(result.frame, figures_dir)
    _plot_interactive_cfgi(result.frame, figures_dir)


def _plot_equity(equity: pd.DataFrame, figures_dir: Path) -> None:

    # Dibuja una línea por estrategia con la evolución del capital.
    axis = equity.plot(figsize=(12, 6), title="Curvas de capital", linewidth=1.4)
    # Títulos de los ejes. El eje X queda sin título porque son fechas.
    axis.set_ylabel("Capital inicial = 100")
    axis.set_xlabel("")

    # Grilla suave de fondo.
    axis.grid(alpha=0.25)

    # Ajusta los márgenes, guarda el PNG en alta resolución y cierra la figura para liberar memoria.
    plt.tight_layout()
    plt.savefig(figures_dir / "equity_curves.png", dpi=180)
    plt.close()


def _plot_drawdown(equity: pd.DataFrame, figures_dir: Path) -> None:

    # Drawdown: capital de cada día sobre el máximo alcanzado hasta ese día, menos 1.
    drawdown = equity / equity.cummax() - 1.0
    # Dibuja una línea por estrategia.
    axis = drawdown.plot(figsize=(12, 5), title="Drawdowns", linewidth=1.2)
    axis.set_ylabel("Drawdown")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    # Ajusta, guarda y cierra.
    plt.tight_layout()
    plt.savefig(figures_dir / "drawdowns.png", dpi=180)
    plt.close()


def _plot_positions(frame: pd.DataFrame, figures_dir: Path) -> None:

    # Posición diaria en BTC de cada estrategia (positiva largo, negativa corto, cero refugio).
    columns = ["strategy_position", "momentum_position"]
    axis = frame[columns].plot(figsize=(12, 5), title="Exposición a BTC", linewidth=1.0)

    axis.set_ylabel("Peso")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    # Ajusta, guarda y cierra.
    plt.tight_layout()
    plt.savefig(figures_dir / "positions.png", dpi=180)
    plt.close()


def _plot_cfgi(frame: pd.DataFrame, figures_dir: Path) -> None:

    # El CFGI y sus cuatro umbrales históricos.
    columns = ["cfgi", "p10", "p20", "p80", "p90"]
    axis = frame[columns].plot(
        figsize=(12, 6),
        title="CFGI y umbrales históricos",
        linewidth=1.0,
    )

    # Sombrea en rojo, de 0 a 100, los días en que la estrategia estuvo en refugio.
    axis.fill_between(
        frame.index,
        0,
        100,
        where=frame["risk_regime_held"].eq(0),
        alpha=0.12,
        color="red",
        label="Refugio",
    )

    # El índice va de 0 a 100.
    axis.set_ylim(0, 100)
    axis.set_ylabel("Índice")
    axis.set_xlabel("")
    axis.grid(alpha=0.25)
    # Leyenda en tres columnas para que no tape el gráfico.
    axis.legend(ncol=3)
    # Ajusta, guarda y cierra.
    plt.tight_layout()
    plt.savefig(figures_dir / "cfgi_regimes.png", dpi=180)
    plt.close()


def _returns_accessor(returns: pd.Series) -> vbt.returns.accessors.ReturnsSRAccessor:
    # Envuelve una serie de retornos diarios en el accesor de vectorbt, con año de 365 días.
    return vbt.returns.accessors.ReturnsSRAccessor(
        returns,
        freq="1D",
        year_freq="365D",
    )


def _write_html(figure: go.Figure, path: Path) -> None:
    # Guarda el gráfico como HTML. La librería de plotly se carga desde internet para que el archivo pese menos.
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def _plot_interactive_equity(returns: pd.DataFrame, figures_dir: Path) -> None:
    # Accesor sobre los retornos de la estrategia con CFGI.
    accessor = _returns_accessor(returns["strategy"])
    # Curva de capital de la estrategia, con el momentum puro como referencia, arrancando en 100.
    figure = accessor.plot_cumulative(
        benchmark_rets=returns["momentum"],
        start_value=100,
        title="Curvas de capital interactivas",
    )
    # Renombra las dos series en la leyenda.
    figure.data[0].name = "Estrategia con CFGI"
    figure.data[1].name = "Momentum"
    figure.update_layout(xaxis_title="", yaxis_title="Capital inicial = 100")
    _write_html(figure, figures_dir / "equity_curves_interactive.html")


def _plot_interactive_drawdowns(returns: pd.DataFrame, figures_dir: Path) -> None:
    # Un gráfico por estrategia.
    for name, title in {
        "strategy": "Drawdowns de la estrategia con CFGI",
        "momentum": "Drawdowns de momentum",
    }.items():
        # Marca los 10 peores drawdowns sobre la curva de capital.
        figure = _returns_accessor(returns[name]).drawdowns.plot(top_n=10, title=title)
        figure.update_layout(xaxis_title="", yaxis_title="Capital acumulado")
        _write_html(figure, figures_dir / f"drawdowns_{name}_interactive.html")


def _plot_interactive_positions(frame: pd.DataFrame, figures_dir: Path) -> None:
    # Figura vacía de plotly.
    figure = go.Figure()
    # Línea con la posición diaria de la estrategia con CFGI.
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["strategy_position"],
            name="Estrategia con CFGI",
            mode="lines",
        )
    )
    # Línea con la posición diaria del momentum puro.
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["momentum_position"],
            name="Momentum",
            mode="lines",
        )
    )
    # hovermode="x unified" muestra los dos valores juntos al pasar el mouse por una fecha.
    figure.update_layout(
        title="Exposición interactiva a BTC",
        xaxis_title="",
        yaxis_title="Peso",
        hovermode="x unified",
    )
    _write_html(figure, figures_dir / "positions_interactive.html")


def _plot_interactive_cfgi(frame: pd.DataFrame, figures_dir: Path) -> None:
    # Figura vacía de plotly.
    figure = go.Figure()
    # Serie que vale 100 los días en refugio y queda vacía el resto, para sombrear solo esos días.
    refuge = frame["risk_regime_held"].eq(0).astype(float).mul(100).replace(0, float("nan"))
    # Área roja sin borde que rellena desde cero hasta 100 en los días de refugio.
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
    # Una línea para el CFGI y una para cada umbral.
    for column, name in {
        "cfgi": "CFGI",
        "p10": "p10",
        "p20": "p20",
        "p80": "p80",
        "p90": "p90",
    }.items():
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=name, mode="lines"))
    # Eje Y fijo de 0 a 100 y todos los valores juntos al pasar el mouse.
    figure.update_layout(
        title="CFGI y umbrales históricos interactivos",
        xaxis_title="",
        yaxis_title="Índice",
        yaxis_range=[0, 100],
        hovermode="x unified",
    )
    _write_html(figure, figures_dir / "cfgi_regimes_interactive.html")
