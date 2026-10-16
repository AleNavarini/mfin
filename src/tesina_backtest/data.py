from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path
from time import sleep

import pandas as pd
import requests

from tesina_backtest.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
CFGI_URL = "https://api.alternative.me/fng/"
DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


REQUEST_TIMEOUT_SECONDS = 30


Downloader = Callable[[Path], pd.DataFrame]


def _get_json(url: str, params: dict[str, str | int] | None = None) -> object:

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def download_btc_usdt_daily(output_path: Path) -> pd.DataFrame:

    rows: list[list[object]] = []

    start_ms = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)

    while True:
        payload = _get_json(
            BINANCE_KLINES_URL,
            params={
                "symbol": "BTCUSDT",
                "interval": "1d",
                "startTime": start_ms,
                "limit": 1000,
            },
        )

        if not isinstance(payload, list) or not payload:
            break

        rows.extend(payload)

        last_open_ms = int(payload[-1][0])

        next_start_ms = last_open_ms + 86_400_000

        if next_start_ms <= start_ms or len(payload) < 1000:
            break

        start_ms = next_start_ms

        sleep(0.1)

    if not rows:
        raise ValueError("Binance no devolvió precios de BTC/USDT")

    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base_volume",
            "taker_quote_volume",
            "ignore",
        ],
    )

    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)

    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)

    frame["close"] = pd.to_numeric(frame["close"], errors="raise")

    now = pd.Timestamp.now(tz="UTC")

    completed = frame.loc[frame["close_time"] <= now, ["date", "close"]]

    result = completed.drop_duplicates("date").set_index("date").sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def download_cfgi(output_path: Path) -> pd.DataFrame:

    payload = _get_json(CFGI_URL, params={"limit": 0, "format": "json"})

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError("Alternative.me devolvió un formato de CFGI inesperado")

    frame = pd.DataFrame(payload["data"])

    frame["date"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"]), unit="s", utc=True
    ).dt.tz_localize(None)

    frame["cfgi"] = pd.to_numeric(frame["value"], errors="raise")

    result = frame[["date", "cfgi"]].drop_duplicates("date").set_index("date").sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def download_dtb3(output_path: Path) -> pd.DataFrame:

    response = requests.get(DTB3_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))

    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    frame["date"] = pd.to_datetime(frame[date_column])

    frame["dtb3"] = pd.to_numeric(frame["DTB3"], errors="coerce")

    result = frame[["date", "dtb3"]].drop_duplicates("date").set_index("date").sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def download_btc_usdt_funding(output_path: Path) -> pd.DataFrame:

    rows: list[dict[str, object]] = []

    start_ms = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)

    while True:
        payload = _get_json(
            BINANCE_FUNDING_URL,
            params={"symbol": "BTCUSDT", "startTime": start_ms, "limit": 1000},
        )

        if not isinstance(payload, list) or not payload:
            break

        rows.extend(payload)

        next_start_ms = int(payload[-1]["fundingTime"]) + 1

        if len(payload) < 1000:
            break

        start_ms = next_start_ms

        sleep(0.1)

    if not rows:
        raise ValueError("Binance no devolvió tasas de funding de BTCUSDT")

    frame = pd.DataFrame(rows)

    funding_time = pd.to_datetime(pd.to_numeric(frame["fundingTime"]), unit="ms", utc=True)

    # El pago de las 00:00 UTC cierra el día anterior: corresponde a la posición
    # mantenida durante ese día, igual que el retorno de cierre a cierre.
    frame["date"] = (funding_time - pd.Timedelta(milliseconds=1)).dt.floor("D").dt.tz_localize(None)

    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="raise")

    result = (
        frame.drop_duplicates("fundingTime").groupby("date")[["funding_rate"]].sum().sort_index()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def _load_or_download(path: Path, downloader: Downloader, refresh: bool) -> pd.DataFrame:

    if refresh or not path.exists():
        return downloader(path)

    return pd.read_csv(path, index_col="date", parse_dates=True)


def load_dataset(refresh: bool = False) -> pd.DataFrame:

    btc = _load_or_download(DATA_RAW_DIR / "btc_usdt_daily.csv", download_btc_usdt_daily, refresh)
    cfgi = _load_or_download(DATA_RAW_DIR / "cfgi_daily.csv", download_cfgi, refresh)
    dtb3 = _load_or_download(DATA_RAW_DIR / "dtb3_daily.csv", download_dtb3, refresh)
    funding = _load_or_download(
        DATA_RAW_DIR / "btc_usdt_funding_daily.csv", download_btc_usdt_funding, refresh
    )

    dataset = btc.join(cfgi, how="inner").join(dtb3, how="left").join(funding, how="left")

    dataset["dtb3"] = dataset["dtb3"].ffill()

    # Antes del lanzamiento del perpetuo (10/09/2019) no existe funding observado.
    dataset["funding_rate"] = dataset["funding_rate"].fillna(0.0)

    if dataset[["close", "cfgi", "dtb3"]].isna().any().any():
        missing = dataset[["close", "cfgi", "dtb3"]].isna().sum().to_dict()
        raise ValueError(f"El dataset alineado contiene valores faltantes: {missing}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PROCESSED_DIR / "daily_dataset.csv")

    return dataset
