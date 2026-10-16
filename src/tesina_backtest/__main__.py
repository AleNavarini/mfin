"""Punto de entrada: python -m tesina_backtest <análisis> [--refresh]."""

import argparse

from tesina_backtest.analyses import base, cfgi_filters, costs, momentum_windows
from tesina_backtest.data import load_dataset

ANALYSES = {
    "base": base.run,
    "ventanas": momentum_windows.run,
    "filtros": cfgi_filters.run,
    "costos": costs.run,
}


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="python -m tesina_backtest", description="Backtests de la tesina de momentum y CFGI"
    )
    parser.add_argument(
        "analysis",
        choices=[*ANALYSES, "todo"],
        help="Análisis a ejecutar. 'todo' corre los cuatro en orden",
    )
    parser.add_argument("--refresh", action="store_true", help="Vuelve a descargar los datos")
    args = parser.parse_args()

    dataset = load_dataset(refresh=args.refresh)

    selected = ANALYSES if args.analysis == "todo" else {args.analysis: ANALYSES[args.analysis]}

    for name, run in selected.items():
        print(f"\n=== {name} ===")
        run(dataset)


if __name__ == "__main__":
    main()
