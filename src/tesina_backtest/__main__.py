"""Punto de entrada: python -m tesina_backtest [--sin-costos] [--refresh]."""

# Librería estándar para leer los argumentos de la línea de comandos.
import argparse

# Parámetros por defecto del backtest.
from tesina_backtest.config import BacktestConfig

# Función que descarga (o lee de disco) y alinea precios, CFGI, DTB3 y funding.
from tesina_backtest.data import load_dataset

# Función que corre todas las configuraciones de la grilla.
from tesina_backtest.grid import run_grid


def main() -> None:

    # Crea el lector de argumentos con el nombre del programa y una descripción para --help.
    parser = argparse.ArgumentParser(
        prog="python -m tesina_backtest", description="Backtests de la tesina de momentum y CFGI"
    )
    # Bandera opcional: si se pasa, los retornos son brutos (sin comisiones, slippage ni funding).
    parser.add_argument("--sin-costos", action="store_true", help="Corre la grilla sin costos")
    # Bandera opcional: si se pasa, fuerza la descarga de datos aunque ya existan en disco.
    parser.add_argument("--refresh", action="store_true", help="Vuelve a descargar los datos")
    # Lee lo que el usuario escribió en la terminal.
    args = parser.parse_args()

    # Carga el dataset diario alineado.
    dataset = load_dataset(refresh=args.refresh)

    # Configuración base con los costos prendidos o apagados.
    base = BacktestConfig(apply_costs=not args.sin_costos)

    # Corre la grilla completa a partir de esa configuración.
    run_grid(dataset, base)


# Solo corre main() cuando el archivo se ejecuta directamente, no cuando se importa.
if __name__ == "__main__":
    main()
