"""Reconstruye Silver SIN Airflow: corre las mismas consultas de include/sql/
contra el database.sqlite original y aplica las mismas funciones de
include/src/transform.py, en el mismo orden que el DAG.

No reemplaza al DAG: sirve para trabajar el notebook de modelado sin levantar
Docker, y para verificar que el Silver del DAG y este son idénticos
(README §15: "es idéntico al que resulta de correr las mismas consultas contra
el SQLite original").

    .venv/Scripts/python -m scripts.build_silver_local
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from include.config import (
    AUDIT_DIR,
    BRONZE_DIR,
    BRONZE_QUERIES,
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    SILVER_CSV_PATH,
    SILVER_DATASET_PATH,
    SOCCER_DB_PATH,
    SQL_DIR,
)
from include.src import profiling, quality_check, storage, transform
from include.src.source_loader import DATE_COLUMNS


def run_query(con: sqlite3.Connection, sql_file: str) -> pd.DataFrame:
    """La consulta de Bronze contra SQLite, con los dos ajustes que hace el seed
    al cargar PostgreSQL: nombres de columna en minúscula y fechas como
    timestamp (source_loader.DATE_COLUMNS)."""
    df = pd.read_sql_query((SQL_DIR / sql_file).read_text(encoding="utf-8"), con)
    df.columns = [col.lower() for col in df.columns]
    for col in DATE_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col])
    return df


def main() -> None:
    with sqlite3.connect(SOCCER_DB_PATH) as con:
        bronze = {name: run_query(con, sql_file) for name, sql_file in BRONZE_QUERIES.items()}

    for name, df in bronze.items():
        storage.write_parquet(df, BRONZE_DIR / f"{name}.parquet")

    prepared, excluded_prepare, _ = transform.prepare_matches(bronze["matches"], bronze["teams"])
    lineups, _ = transform.build_lineup_features(prepared, bronze["players"], bronze["player_attributes"])
    with_target = transform.define_underdog_and_target(prepared)
    valid, excluded_filter, _ = transform.filter_valid_underdogs(with_target)
    silver, audit = transform.build_silver(valid, lineups)

    # las mismas capas intermedias que persiste el DAG (las leen los tests y el notebook de la Entrega 2)
    for df, name in [(prepared, "matches_prepared"), (lineups, "lineup_features"),
                     (with_target, "matches_with_target"), (valid, "matches_valid")]:
        storage.write_parquet(df, INTERMEDIATE_DIR / f"{name}.parquet")
    storage.write_csv(profiling.threshold_sensitivity(with_target), REPORTS_DIR / "threshold_sensitivity.csv")

    storage.write_parquet(audit, AUDIT_DIR / "match_audit.parquet")
    storage.write_parquet(
        pd.concat([excluded_prepare, excluded_filter], ignore_index=True), AUDIT_DIR / "excluded_matches.parquet"
    )
    storage.write_csv(silver, SILVER_CSV_PATH)
    storage.write_parquet(silver, SILVER_DATASET_PATH)

    report = quality_check.run_quality_checks(silver)
    quality_check.assert_quality(report)
    print(f"Silver: {silver.shape[0]} filas x {silver.shape[1]} columnas -> {SILVER_DATASET_PATH}")
    print(f"Quality checks: {len(report['hard_checks'])} hard checks OK")


if __name__ == "__main__":
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    main()
