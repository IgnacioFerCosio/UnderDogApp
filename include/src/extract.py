"""Capa Bronze: el resultado de cada consulta SQL a source-db, persistido sin
transformar.

Bronze NO es el SQLite descargado: es lo que devolvió la base cuando Airflow la
consultó. Cada archivo tiene exactamente las columnas, el orden y los valores
que entregó el driver (los enteros siguen siendo enteros, los nulos siguen
siendo nulos, las fechas siguen siendo timestamps). Ninguna regla de negocio
se aplica acá.

Se guardan dos copias por tabla:
- `bronze/<tabla>.parquet`: la última corrida, lo que lee el resto del DAG.
- `bronze/<tabla>/<version>/bronze.parquet` + `meta.json`: histórico, una
  carpeta por cada huella de fuente+código distinta (`version`, la huella que
  arma `include/src/change_detection.py`). `meta.json` deja ese snapshot
  legible a simple vista (filas, columnas, query, cuándo se extrajo) sin tener
  que ir a los logs de esa corrida. Si la huella no cambió desde la última vez
  que se llegó hasta acá, la carpeta ya existe: no se vuelve a consultar la
  fuente, solo se refresca la copia de "última corrida".
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from include.config import BRONZE_DIR, BRONZE_QUERIES
from include.src.database import QueryResult, fetch_from_source, read_sql
from include.src.storage import write_arrow_table, write_json

logger = logging.getLogger(__name__)


def query_result_to_arrow(result: QueryResult) -> pa.Table:
    """Convierte el resultado de la consulta a una tabla Arrow columna por
    columna, dejando que Arrow infiera el tipo desde los objetos Python que
    devolvió el driver. No pasa por pandas, que convertiría a float los
    enteros con nulos."""
    if not result.rows:
        raise ValueError("La consulta no devolvió filas: no hay nada para escribir en Bronze.")
    values_by_column = zip(*result.rows)
    arrays = [pa.array(list(values)) for values in values_by_column]
    return pa.Table.from_arrays(arrays, names=result.columns)


def extract_to_bronze(table_name: str, version: str) -> dict:
    """Consulta source-db con `include/sql/<consulta>.sql` y escribe Bronze.

    `version` es la huella de fuente+código de esta corrida (ver
    `include/src/change_detection.py`): identifica la carpeta del histórico.
    Si ya existe una con esa huella, se reusa en vez de volver a consultar la
    fuente, y solo se refresca la copia plana `bronze/<tabla>.parquet`.

    Devuelve metadata chica (ruta, filas, columnas, hashes) para XCom.
    """
    sql_file = BRONZE_QUERIES[table_name]
    query_sha256 = hashlib.sha256(read_sql(sql_file).encode("utf-8")).hexdigest()
    version_tag = version[:16]
    history_dir = BRONZE_DIR / table_name / version_tag
    history_path = history_dir / "bronze.parquet"
    latest_path = BRONZE_DIR / f"{table_name}.parquet"

    if history_path.exists():
        logger.info("Bronze %s: version %s ya está en el histórico, no se vuelve a consultar la fuente.", table_name, version_tag)
        table = pq.read_table(history_path)
    else:
        result = fetch_from_source(sql_file)
        table = query_result_to_arrow(result)
        history_meta = write_arrow_table(table, history_path)
        write_json(
            {
                **history_meta,
                "table": table_name,
                "version": version_tag,
                "query": sql_file,
                "query_sha256": query_sha256,
                "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            history_dir / "meta.json",
        )

    meta = write_arrow_table(table, latest_path)
    meta["table"] = table_name
    meta["version"] = version_tag
    meta["query"] = sql_file
    meta["query_sha256"] = query_sha256
    logger.info(
        "Bronze %s: %d filas x %d columnas -> %s (histórico: %s)", table_name, meta["rows"], meta["columns"], meta["path"], history_path
    )
    return meta
