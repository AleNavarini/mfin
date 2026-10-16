# Permite usar anotaciones de tipos modernas (por ejemplo dict[str, int]) en cualquier versión.
from __future__ import annotations

# Callable describe el tipo "función" en las anotaciones.
from collections.abc import Callable

# StringIO permite leer un texto en memoria como si fuera un archivo.
from io import StringIO

# Path maneja rutas de archivos.
from pathlib import Path

# sleep pausa la ejecución para no saturar las APIs con pedidos seguidos.
from time import sleep

# pandas maneja las tablas de datos (DataFrame) y las series de tiempo (Series).
import pandas as pd

# requests hace los pedidos HTTP a las APIs.
import requests

# Carpetas donde se guardan los datos crudos y el dataset procesado.
from tesina_backtest.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

# Velas diarias de BTC/USDT en el mercado de contado de Binance.
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
# Historial de tasas de funding del perpetuo BTCUSDT de Binance.
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
# Historial del Crypto Fear & Greed Index de Alternative.me.
CFGI_URL = "https://api.alternative.me/fng/"
# Rendimiento de la letra del Tesoro de EE.UU. a 3 meses (DTB3) publicado por FRED.
DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


# Segundos máximos de espera por cada pedido HTTP antes de dar error.
REQUEST_TIMEOUT_SECONDS = 30


# Tipo de las funciones de descarga: reciben la ruta de salida y devuelven un DataFrame.
Downloader = Callable[[Path], pd.DataFrame]


def _get_json(url: str, params: dict[str, str | int] | None = None) -> object:

    # Hace el pedido GET con los parámetros de consulta y el tiempo máximo de espera.
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    # Si la respuesta es un error HTTP (4xx o 5xx), lanza una excepción.
    response.raise_for_status()
    # Convierte el cuerpo JSON de la respuesta en listas y diccionarios de Python.
    return response.json()


def download_btc_usdt_daily(output_path: Path) -> pd.DataFrame:

    # Acumula todas las velas que devuelve Binance, página por página.
    rows: list[list[object]] = []

    # Fecha desde la que se pide historia, en milisegundos desde 1970 como exige Binance.
    start_ms = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)

    # Binance devuelve como máximo 1000 velas por pedido, así que se pagina hasta el final.
    while True:
        # Pide hasta 1000 velas diarias de BTCUSDT desde start_ms.
        payload = _get_json(
            BINANCE_KLINES_URL,
            params={
                "symbol": "BTCUSDT",
                "interval": "1d",
                "startTime": start_ms,
                "limit": 1000,
            },
        )

        # Si la respuesta no es una lista o viene vacía, no hay más datos.
        if not isinstance(payload, list) or not payload:
            break

        # Agrega las velas de esta página al acumulado.
        rows.extend(payload)

        # Hora de apertura de la última vela recibida (primer campo de cada vela).
        last_open_ms = int(payload[-1][0])

        # La siguiente página arranca un día después (86.400.000 ms = 24 horas).
        next_start_ms = last_open_ms + 86_400_000

        # Corta si no avanzó en el tiempo o si la página vino incompleta (ya es la última).
        if next_start_ms <= start_ms or len(payload) < 1000:
            break

        # Avanza el inicio para el próximo pedido.
        start_ms = next_start_ms

        # Pausa de 0,1 segundos para respetar los límites de la API.
        sleep(0.1)

    # Si no llegó ninguna vela, algo falló en la descarga.
    if not rows:
        raise ValueError("Binance no devolvió precios de BTC/USDT")

    # Arma la tabla con los nombres de los 12 campos que devuelve Binance por vela.
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

    # Convierte la hora de apertura (ms) en fecha UTC y le quita la zona horaria.
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)

    # Convierte la hora de cierre (ms) en fecha UTC para comparar con la hora actual.
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)

    # Binance devuelve los precios como texto. Los pasa a número y falla si alguno no es válido.
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")

    # Hora actual en UTC.
    now = pd.Timestamp.now(tz="UTC")

    # Se queda solo con las velas ya cerradas (descarta la del día en curso) y con fecha y cierre.
    completed = frame.loc[frame["close_time"] <= now, ["date", "close"]]

    # Elimina fechas repetidas, usa la fecha como índice y ordena de la más vieja a la más nueva.
    result = completed.drop_duplicates("date").set_index("date").sort_index()

    # Crea la carpeta de salida si no existe.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Guarda los precios en CSV para no tener que descargarlos de nuevo.
    result.to_csv(output_path)
    return result


def download_cfgi(output_path: Path) -> pd.DataFrame:

    # limit=0 pide la historia completa del índice en un solo pedido.
    payload = _get_json(CFGI_URL, params={"limit": 0, "format": "json"})

    # Verifica que la respuesta tenga la forma esperada: un diccionario con una lista en "data".
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError("Alternative.me devolvió un formato de CFGI inesperado")

    # Cada elemento de "data" es un día con "value" y "timestamp".
    frame = pd.DataFrame(payload["data"])

    # El timestamp viene en segundos desde 1970 como texto. Lo pasa a fecha sin zona horaria.
    frame["date"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"]), unit="s", utc=True
    ).dt.tz_localize(None)

    # El valor del índice (0 a 100) viene como texto. Lo pasa a número.
    frame["cfgi"] = pd.to_numeric(frame["value"], errors="raise")

    # Se queda con fecha y valor, quita duplicados, indexa por fecha y ordena.
    result = frame[["date", "cfgi"]].drop_duplicates("date").set_index("date").sort_index()
    # Crea la carpeta y guarda el CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def download_dtb3(output_path: Path) -> pd.DataFrame:

    # FRED devuelve directamente un CSV, no un JSON.
    response = requests.get(DTB3_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    # Lanza una excepción si la respuesta es un error HTTP.
    response.raise_for_status()

    # Lee el texto del CSV como si fuera un archivo.
    frame = pd.read_csv(StringIO(response.text))

    # FRED cambió el nombre de la columna de fecha con el tiempo. Acepta los dos nombres.
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    frame["date"] = pd.to_datetime(frame[date_column])

    # Los feriados vienen como "." en FRED. errors="coerce" los convierte en valores vacíos (NaN).
    frame["dtb3"] = pd.to_numeric(frame["DTB3"], errors="coerce")

    # Se queda con fecha y tasa, quita duplicados, indexa por fecha y ordena.
    result = frame[["date", "dtb3"]].drop_duplicates("date").set_index("date").sort_index()
    # Crea la carpeta y guarda el CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def download_btc_usdt_funding(output_path: Path) -> pd.DataFrame:

    # Acumula todos los pagos de funding, página por página.
    rows: list[dict[str, object]] = []

    # Pide desde antes del lanzamiento del perpetuo para no perder ningún pago.
    start_ms = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)

    # Binance devuelve como máximo 1000 pagos por pedido, así que se pagina hasta el final.
    while True:
        # Pide hasta 1000 pagos de funding desde start_ms.
        payload = _get_json(
            BINANCE_FUNDING_URL,
            params={"symbol": "BTCUSDT", "startTime": start_ms, "limit": 1000},
        )

        # Si no hay respuesta válida o viene vacía, termina.
        if not isinstance(payload, list) or not payload:
            break

        # Agrega los pagos de esta página al acumulado.
        rows.extend(payload)

        # La siguiente página arranca 1 ms después del último pago recibido.
        next_start_ms = int(payload[-1]["fundingTime"]) + 1

        # Una página incompleta significa que ya es la última.
        if len(payload) < 1000:
            break

        # Avanza el inicio para el próximo pedido.
        start_ms = next_start_ms

        # Pausa para respetar los límites de la API.
        sleep(0.1)

    # Si no llegó ningún pago, algo falló en la descarga.
    if not rows:
        raise ValueError("Binance no devolvió tasas de funding de BTCUSDT")

    # Cada fila es un pago con "fundingTime" y "fundingRate".
    frame = pd.DataFrame(rows)

    # Hora exacta de cada pago (00:00, 08:00 y 16:00 UTC), convertida desde milisegundos.
    funding_time = pd.to_datetime(pd.to_numeric(frame["fundingTime"]), unit="ms", utc=True)

    # El pago de las 00:00 UTC cierra el día anterior: corresponde a la posición
    # mantenida durante ese día, igual que el retorno de cierre a cierre.
    frame["date"] = (funding_time - pd.Timedelta(milliseconds=1)).dt.floor("D").dt.tz_localize(None)

    # La tasa viene como texto. La pasa a número.
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="raise")

    # Quita pagos repetidos y suma los tres pagos de cada día para obtener la tasa diaria.
    result = (
        frame.drop_duplicates("fundingTime").groupby("date")[["funding_rate"]].sum().sort_index()
    )

    # Crea la carpeta y guarda el CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def _load_or_download(path: Path, downloader: Downloader, refresh: bool) -> pd.DataFrame:

    # Descarga si se pidió refrescar o si todavía no existe el archivo en disco.
    if refresh or not path.exists():
        return downloader(path)

    # Si ya existe, lo lee del CSV con la fecha como índice.
    return pd.read_csv(path, index_col="date", parse_dates=True)


def load_dataset(refresh: bool = False) -> pd.DataFrame:

    # Carga (o descarga) cada una de las cuatro fuentes por separado.
    btc = _load_or_download(DATA_RAW_DIR / "btc_usdt_daily.csv", download_btc_usdt_daily, refresh)
    cfgi = _load_or_download(DATA_RAW_DIR / "cfgi_daily.csv", download_cfgi, refresh)
    dtb3 = _load_or_download(DATA_RAW_DIR / "dtb3_daily.csv", download_dtb3, refresh)
    funding = _load_or_download(
        DATA_RAW_DIR / "btc_usdt_funding_daily.csv", download_btc_usdt_funding, refresh
    )

    # Une todo por fecha. Inner con el CFGI: solo quedan días con precio y con índice.
    # Left con DTB3 y funding: se mantienen todos esos días aunque falte el dato.
    dataset = btc.join(cfgi, how="inner").join(dtb3, how="left").join(funding, how="left")

    # DTB3 no se publica fines de semana ni feriados. Se completa con el último valor conocido.
    dataset["dtb3"] = dataset["dtb3"].ffill()

    # Antes del lanzamiento del perpetuo (10/09/2019) no existe funding observado.
    dataset["funding_rate"] = dataset["funding_rate"].fillna(0.0)

    # Control de calidad: precio, CFGI y DTB3 no pueden tener huecos después de alinear.
    if dataset[["close", "cfgi", "dtb3"]].isna().any().any():
        # Cuenta los faltantes por columna para mostrarlos en el error.
        missing = dataset[["close", "cfgi", "dtb3"]].isna().sum().to_dict()
        raise ValueError(f"El dataset alineado contiene valores faltantes: {missing}")

    # Guarda el dataset alineado para poder revisarlo.
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PROCESSED_DIR / "daily_dataset.csv")

    return dataset
