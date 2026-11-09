"""Genera la aplicación de la Entrega 4: docs/index.html (+ docs/data/partidos.json).

Toma lo que produce el pipeline (Silver y auditoría del DAG) y lo que dejó el notebook de modelado
(include/data/models/underdog_modelo.json y resultados_e3.json), y arma una página estática con gráficos
de Altair. GitHub Pages la publica desde la carpeta docs/ del repositorio.

    .venv/Scripts/python app/build_app.py

Si los datos cambian: correr el DAG, re-ejecutar el notebook y correr este script de nuevo.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from include.config import AUDIT_DIR, DATA_DIR, SILVER_DATASET_PATH, SILVER_MANIFEST_PATH  # noqa: E402
from include.src.columns import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN  # noqa: E402
from include.src.features import logit  # noqa: E402

SALIDA = RAIZ / "docs"
MODELOS_DIR = DATA_DIR / "models"
TEST_TEMPORADA = "2015/2016"
APUESTA = 100

# Versiones exactas de la librería que dibuja los gráficos de Altair en el navegador
VEGA, VEGA_LITE, VEGA_EMBED = "6.2.0", "6.4.1", "7.0.2"

# Los mismos colores que el notebook, con el mismo significado en toda la aplicación
COLOR = {"modelo": "#0f8b7d", "mercado": "#e08a1e", "todos": "#c0392b", "gano": "#2563a8", "no_gano": "#b9bec7"}
RESULTADO = alt.Scale(domain=["Ganó el underdog", "No ganó"], range=[COLOR["gano"], COLOR["no_gano"]])
ESTRATEGIA = alt.Scale(domain=["Modelo (apuesta si vale la pena)", "Apostar a todos los underdogs", "Mercado como modelo (no apuesta)"],
                       range=[COLOR["modelo"], COLOR["todos"], COLOR["mercado"]])

NOMBRES_GAPS = {
    "nofav_xi_overall_mean_gap": "Rating medio del XI", "nofav_top3_overall_gap": "Rating de los 3 mejores",
    "nofav_best_overall_gap": "Mejor jugador", "nofav_worst_overall_gap": "Peor titular",
    "nofav_def_overall_gap": "Rating de la defensa", "nofav_mid_overall_gap": "Rating del mediocampo",
    "nofav_att_overall_gap": "Rating del ataque", "nofav_best_finishing_gap": "Mejor definición",
    "nofav_best_reactions_gap": "Mejores reacciones", "nofav_best_marking_gap": "Mejor marca",
}
LIGAS = {
    "England Premier League": "Inglaterra", "France Ligue 1": "Francia", "Germany 1. Bundesliga": "Alemania",
    "Italy Serie A": "Italia", "Spain LIGA BBVA": "España", "Netherlands Eredivisie": "Países Bajos",
    "Portugal Liga ZON Sagres": "Portugal", "Scotland Premier League": "Escocia", "Belgium Jupiler League": "Bélgica",
}


# ----------------------------------------------------------------------------- datos y modelo
def cargar() -> tuple[pd.DataFrame, dict, dict]:
    ficha = json.loads((MODELOS_DIR / "underdog_modelo.json").read_text(encoding="utf-8"))
    resultados = json.loads((MODELOS_DIR / "resultados_e3.json").read_text(encoding="utf-8"))
    df = (pd.read_parquet(SILVER_DATASET_PATH)
            .merge(pd.read_parquet(AUDIT_DIR / "match_audit.parquet"), on="match_id")
            .sort_values(["fecha", "match_id"]).reset_index(drop=True))
    df["cuota_underdog"] = np.where(df["equipo_favorito"] == "local", df["odds_away"], df["odds_home"])
    df["prob_modelo"] = predecir(df, ficha)
    return df, ficha, resultados


def predecir(df: pd.DataFrame, ficha: dict) -> np.ndarray:
    """La fórmula del modelo, tal cual la escribió el notebook (y la misma que corre la página en JavaScript)."""
    z = np.full(len(df), ficha["intercepto"])
    for col, v in ficha["probabilidades"].items():
        z += v["coef"] * logit(df[col].fillna(v["mediana"]))
    for col, v in ficha["numericas"].items():
        z += v["coef"] * (df[col].fillna(v["mediana"]) - v["media"]) / v["desvio"]
    return 1 / (1 + np.exp(-z))


def verificar_contra_el_pipeline(df: pd.DataFrame) -> float:
    """Lo que muestra la app tiene que coincidir con el modelo del notebook."""
    import joblib
    modelo = joblib.load(MODELOS_DIR / "underdog_modelo.joblib")
    return float(np.max(np.abs(modelo.predict_proba(df)[:, 1] - df["prob_modelo"].values)))


def fuera_de_muestra(df: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    """Los partidos que se predijeron sin haberlos visto, con esa predicción en `p`:
    la validación temporal (cada uno lo predice un modelo entrenado sólo con los anteriores, exportado por el notebook)
    y el test (el modelo final, que nunca vio 2015/16). Los de 2008–2010 sólo sirvieron para entrenar y quedan afuera."""
    oof = pd.Series(list(resultados["oof_modelo"].values()), index=resultados["oof_indices_match_id"])
    val = df[df["match_id"].isin(oof.index)].assign(p=lambda d: d["match_id"].map(oof), conjunto="validación temporal (2010–2015)")
    test = df[df["temporada"] == TEST_TEMPORADA].assign(p=lambda d: d["prob_modelo"], conjunto="test 2015/16 (modelo final)")
    return pd.concat([val, test]).sort_values(["fecha", "match_id"]).reset_index(drop=True)


def partidos_para_explorar(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por partido fuera de muestra, con lo que habría dicho y hecho el modelo con lo que sabía hasta ese momento."""
    favorito_local = df["equipo_favorito"] == "local"
    valor = df["p"] * df["cuota_underdog"] - 1
    apuesta = valor > 0
    gano = df[TARGET_COLUMN].astype(int)
    return pd.DataFrame({
        "fecha": df["fecha"].dt.strftime("%Y-%m-%d"),
        "temporada": df["temporada"],
        "liga": df["liga"].map(LIGAS).fillna(df["liga"]),
        "conjunto": df["conjunto"],
        "favorito": np.where(favorito_local, df["equipo_local"], df["equipo_visitante"]),
        "underdog": np.where(favorito_local, df["equipo_visitante"], df["equipo_local"]),
        "partido": df["equipo_local"] + " " + df["goles_local"].astype(str) + "–" + df["goles_visitante"].astype(str) + " " + df["equipo_visitante"],
        "resultado": np.where(gano == 1, "Ganó el underdog", "No ganó"),
        "gano": gano,
        "prob_mercado": df["prob_no_favorito"].round(4),
        "prob_modelo": df["p"].round(4),
        "cuota": df["cuota_underdog"].round(2),
        "valor": valor.round(4),
        "decision": np.where(apuesta, "Apuesta", "No apuesta"),
        "apuesta": apuesta.astype(int),
        "ganancia": np.where(apuesta, APUESTA * (df["cuota_underdog"] * gano - 1), 0).round(1),
        "gap_xi": df["nofav_xi_overall_mean_gap"].round(2),
    })


# ----------------------------------------------------------------------------- gráficos del tallo
def grafico_datos(df: pd.DataFrame) -> tuple[alt.Chart, dict]:
    cortes = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    etiquetas = ["hasta 40%", "40–50%", "50–60%", "60–70%", "70–80%", "más de 80%"]
    t = (df.assign(tramo=pd.cut(df["prob_favorito"], cortes, labels=etiquetas))
           .groupby("tramo", observed=True)[TARGET_COLUMN].agg(tasa="mean", partidos="size").reset_index())
    base = alt.Chart(t).encode(
        x=alt.X("tramo:O", sort=etiquetas, title="Probabilidad de ganar del favorito, según la cuota", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("tasa:Q", title="Partidos que ganó el underdog", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.35])),
        tooltip=[alt.Tooltip("tramo:O", title="Favorito"), alt.Tooltip("tasa:Q", title="Gana el underdog", format=".1%"),
                 alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    barras = base.mark_bar(color=COLOR["gano"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    texto = base.mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(text=alt.Text("tasa:Q", format=".0%"))
    numeros = {"alto": t["tasa"].iloc[0], "bajo": t["tasa"].iloc[-1]}
    return (barras + texto).properties(width="container", height=300), numeros


def grafico_hallazgo(df: pd.DataFrame) -> tuple[alt.Chart, dict]:
    q = pd.qcut(df["nofav_xi_overall_mean_gap"], 5)
    etiquetas = [f"{i.left:+.1f} a {i.right:+.1f}" for i in q.cat.categories]
    etiquetas[0] = f"{df['nofav_xi_overall_mean_gap'].min():+.1f} a {q.cat.categories[0].right:+.1f}"
    t = (df.assign(quintil=q.cat.rename_categories(etiquetas))
           .groupby("quintil", observed=True)
           .agg(**{"Victorias reales del underdog": (TARGET_COLUMN, "mean"),
                   "Lo que esperaba la cuota": ("prob_no_favorito", "mean"), "partidos": ("match_id", "size")})
           .reset_index().melt(id_vars=["quintil", "partidos"], var_name="serie", value_name="valor"))
    escala = alt.Scale(domain=["Victorias reales del underdog", "Lo que esperaba la cuota"], range=[COLOR["gano"], COLOR["mercado"]])
    lineas = alt.Chart(t).mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3).encode(
        x=alt.X("quintil:O", sort=etiquetas, title="Rating medio del XI: underdog − favorito (cinco grupos de igual tamaño)",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("valor:Q", title="Proporción", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.32])),
        color=alt.Color("serie:N", scale=escala, title=None, legend=alt.Legend(orient="top-left")),
        strokeDash=alt.condition(alt.datum.serie == "Lo que esperaba la cuota", alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("quintil:O", title="Diferencia de rating"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", title="Valor", format=".1%"), alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    real = t[t["serie"] == "Victorias reales del underdog"]["valor"]
    return lineas.properties(width="container", height=300), {"bajo": real.iloc[0], "alto": real.iloc[-1]}


def grafico_modelo(d: pd.DataFrame) -> alt.Chart:
    """Plata acumulada de cada estrategia, sólo sobre partidos predichos sin haberlos visto (ver fuera_de_muestra)."""
    d = d.copy()
    test = d[d["temporada"] == TEST_TEMPORADA]
    gano = d[TARGET_COLUMN].astype(int)
    d["Apostar a todos los underdogs"] = APUESTA * (d["cuota_underdog"] * gano - 1)
    d["Modelo (apuesta si vale la pena)"] = np.where(d["p"] * d["cuota_underdog"] - 1 > 0, d["Apostar a todos los underdogs"], 0)
    d["Mercado como modelo (no apuesta)"] = 0.0
    estr = list(ESTRATEGIA.domain)
    diario = d.groupby("fecha")[estr].sum().cumsum().div(1000).reset_index()
    largo = diario.melt(id_vars="fecha", var_name="estrategia", value_name="ganancia")
    largo["fecha"] = largo["fecha"].dt.strftime("%Y-%m-%d")
    lineas = alt.Chart(largo).mark_line(strokeWidth=2.5).encode(
        x=alt.X("fecha:T", title="Fecha del partido (siempre predicho por un modelo que no lo había visto)",
                axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("ganancia:Q", title="Ganancia acumulada (miles de $, apostando $100 por partido)"),
        color=alt.Color("estrategia:N", scale=ESTRATEGIA, title=None, legend=alt.Legend(orient="bottom-left")),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("estrategia:N", title="Estrategia"),
                 alt.Tooltip("ganancia:Q", title="Acumulado (miles de $)", format=",.1f")])
    inicio_test = pd.DataFrame({"fecha": [test["fecha"].min().strftime("%Y-%m-%d")], "texto": ["empieza el test 2015/16"]})
    regla = alt.Chart(inicio_test).mark_rule(strokeDash=[3, 3], color="#555").encode(x="fecha:T")
    rotulo = alt.Chart(inicio_test.assign(y=-60)).mark_text(align="right", dx=-6, color="#555").encode(
        x="fecha:T", y="y:Q", text="texto:N")
    return (lineas + regla + rotulo).properties(width="container", height=320)


# ----------------------------------------------------------------------------- la copa
def grafico_explorar(ligas: list[str], temporadas: list[str]) -> alt.Chart:
    datos = alt.UrlData("data/partidos.json")
    liga = alt.param(name="liga", value=None, bind=alt.binding_select(options=[None] + ligas, labels=["Todas"] + ligas, name="Liga  "))
    temporada = alt.param(name="temporada", value=None,
                          bind=alt.binding_select(options=[None] + temporadas, labels=["Todas"] + temporadas, name="Temporada  "))
    zona = alt.selection_interval(name="zona", encodings=["x", "y"])
    filtro = "(liga == null || datum.liga == liga) && (temporada == null || datum.temporada == temporada)"

    puntos = alt.Chart(datos).mark_circle(size=14, opacity=0.45).encode(
        x=alt.X("prob_mercado:Q", title="Probabilidad del underdog según la cuota", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.5])),
        y=alt.Y("valor:Q", title="Valor esperado de apostarle (según el modelo)", axis=alt.Axis(format="+%"),
                scale=alt.Scale(domain=[-0.4, 0.3])),
        color=alt.Color("resultado:N", scale=RESULTADO, title="Qué pasó", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("liga:N", title="Liga"),
                 alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("favorito:N", title="Favorito"),
                 alt.Tooltip("underdog:N", title="Underdog"), alt.Tooltip("cuota:Q", title="Cuota del underdog", format=".2f"),
                 alt.Tooltip("prob_mercado:Q", title="Prob. según la cuota", format=".1%"),
                 alt.Tooltip("prob_modelo:Q", title="Prob. según el modelo", format=".1%"),
                 alt.Tooltip("valor:Q", title="Valor esperado", format="+.1%"), alt.Tooltip("decision:N", title="El modelo"),
                 alt.Tooltip("gap_xi:Q", title="Dif. rating del XI", format="+.1f"), alt.Tooltip("conjunto:N", title="Conjunto")],
    ).add_params(liga, temporada, zona).transform_filter(filtro).properties(width=760, height=380)
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=COLOR["modelo"], strokeDash=[5, 4]).encode(y="y:Q")
    rotulo = alt.Chart(pd.DataFrame({"y": [0], "t": ["por encima de esta línea, el modelo apuesta"]})).mark_text(
        align="left", x=6, dy=-8, color=COLOR["modelo"], fontWeight="bold").encode(y="y:Q", text="t:N")

    plata = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_calculate(
        todos=f"{APUESTA} * (datum.cuota * datum.gano - 1)"
    ).transform_aggregate(
        partidos="count()", suma_todos="sum(todos)", suma_modelo="sum(ganancia)", apuestas="sum(apuesta)"
    ).transform_calculate(**{
        "Apostar a todos los underdogs": f"100 * datum.suma_todos / ({APUESTA} * datum.partidos)",
        "Modelo (apuesta si vale la pena)": f"datum.apuestas > 0 ? 100 * datum.suma_modelo / ({APUESTA} * datum.apuestas) : 0",
    }).transform_fold(["Modelo (apuesta si vale la pena)", "Apostar a todos los underdogs"], as_=["estrategia", "roi"])
    barras = plata.mark_bar(height=26).encode(
        y=alt.Y("estrategia:N", title=None, sort=list(ESTRATEGIA.domain), axis=alt.Axis(labelLimit=260)),
        x=alt.X("roi:Q", title="Retorno por cada $100 apostados (%), en lo que está filtrado",
                scale=alt.Scale(domain=[-40, 40], clamp=True)),
        color=alt.Color("estrategia:N", scale=ESTRATEGIA, legend=None),
        tooltip=[alt.Tooltip("estrategia:N", title="Estrategia"), alt.Tooltip("roi:Q", title="Retorno %", format="+.1f"),
                 alt.Tooltip("partidos:Q", title="Partidos filtrados", format=","),
                 alt.Tooltip("apuestas:Q", title="Apuestas del modelo", format=",")])
    texto = plata.mark_text(align="left", dx=6, fontWeight="bold").encode(
        y=alt.Y("estrategia:N", sort=list(ESTRATEGIA.domain)), x=alt.X("roi:Q", scale=alt.Scale(domain=[-40, 40], clamp=True)),
        text=alt.Text("roi:Q", format="+.1f"))
    resumen = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_aggregate(
        partidos="count()", apuestas="sum(apuesta)").transform_calculate(
        t="format(datum.partidos, ',') + ' partidos filtrados · el modelo apuesta en ' + format(datum.apuestas, ',')"
    ).mark_text(align="left", fontSize=13, color="#333").encode(text="t:N").properties(width=760, height=18)
    cero_roi = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#555").encode(x="x:Q")
    return alt.vconcat((puntos + cero + rotulo), resumen,
                       (barras + texto + cero_roi).properties(width=760, height=90)).resolve_scale(color="independent")


def grafico_probar() -> alt.Chart:
    # fuentes con nombre y sin valores: la página les carga el caso que arma el usuario (view.data en JavaScript)
    caso = alt.NamedData(name="caso")
    escala = alt.Scale(domain=["Según la cuota", "Según el modelo"], range=[COLOR["mercado"], COLOR["modelo"]])
    barras = alt.Chart(caso).mark_bar(height=34).encode(
        y=alt.Y("quien:N", title=None, sort=["Según la cuota", "Según el modelo"]),
        x=alt.X("p:Q", title="Probabilidad de que gane el underdog", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.6])),
        color=alt.Color("quien:N", scale=escala, legend=None),
        tooltip=[alt.Tooltip("quien:N", title=""), alt.Tooltip("p:Q", title="Probabilidad", format=".1%")])
    texto = barras.mark_text(align="left", dx=6, fontWeight="bold", color="#222").encode(text=alt.Text("p:Q", format=".1%"))
    equilibrio = alt.NamedData(name="equilibrio")
    regla = alt.Chart(equilibrio).mark_rule(color=COLOR["todos"], strokeDash=[5, 3], strokeWidth=2).encode(x="p:Q")
    rotulo = alt.Chart(equilibrio).mark_text(align="left", dx=5, dy=-48, color=COLOR["todos"]).encode(
        x="p:Q", text=alt.value("lo que paga la cuota (1 / cuota)"))
    return (barras + texto + regla + rotulo).properties(width="container", height=120)


# ----------------------------------------------------------------------------- tablas de los «Cómo lo sabemos»
def tabla_html(registros: list[dict], columnas: list[str] | None = None, formatos: dict | None = None) -> str:
    formatos = formatos or {}
    columnas = columnas or list(registros[0])
    filas = []
    for r in registros:
        celdas = []
        for c in columnas:
            v = r.get(c)
            if c in formatos and v not in (None, "", "nan"):
                try:
                    v = formatos[c].format(float(v))
                except (TypeError, ValueError):
                    pass
            celdas.append(f"<td>{html.escape('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))}</td>")
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    cab = "".join(f"<th>{html.escape('' if c == 'fila' else c)}</th>" for c in columnas)
    return f"<div class='tbl'><table><tr>{cab}</tr>{''.join(filas)}</table></div>"


def embudo() -> str:
    ex = pd.read_parquet(AUDIT_DIR / "excluded_matches.parquet")["motivo"].value_counts()
    pasos = [("liga_sin_cuotas", "Se sacan las 2 ligas sin cuotas (Polonia y Suiza)"),
             ("alineacion_incompleta", "Se exige la alineación titular completa (22 jugadores)"),
             ("sin_cuotas_1x2", "Se exigen cuotas 1X2 completas de al menos una casa"),
             ("cuota_empate_incoherente", "Se sacan cuotas incoherentes (empate implícito > 40%)"),
             ("sin_underdog_identificable", "Se sacan partidos con las dos probabilidades idénticas"),
             ("partido_parejo", "Se exige un underdog claro: diferencia de probabilidad > 0,05")]
    quedan = 25_979
    filas = [{"paso": "Partidos en la fuente (European Soccer Database)", "se sacan": "", "quedan": f"{quedan:,}".replace(",", ".")}]
    for motivo, texto in pasos:
        n = int(ex.get(motivo, 0))
        quedan -= n
        filas.append({"paso": texto, "se sacan": f"{n:,}".replace(",", "."), "quedan": f"{quedan:,}".replace(",", ".")})
    return tabla_html(filas)


# ----------------------------------------------------------------------------- casos para «Probá el modelo»
def casos_ejemplo(df: pd.DataFrame) -> list[dict]:
    test = df[df["temporada"] == TEST_TEMPORADA].copy()
    test["valor"] = test["prob_modelo"] * test["cuota_underdog"] - 1
    sorpresa = test.loc[test[TARGET_COLUMN] == 1, "prob_modelo"].idxmin()   # la sorpresa más grande del test
    elegidos = pd.concat([test.nlargest(3, "valor"), test.loc[[sorpresa]], test.sample(6, random_state=42)]).drop_duplicates("match_id")
    casos = []
    for _, r in elegidos.iterrows():
        casos.append({
            "nombre": f"{r['equipo_local']} – {r['equipo_visitante']} ({r['fecha']:%d/%m/%Y})",
            "resultado": f"Terminó {r['goles_local']}–{r['goles_visitante']}",
            "cuotas": [float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"])],
            "gaps": {c: (None if pd.isna(r[c]) else round(float(r[c]), 2)) for c in NOMBRES_GAPS},
        })
    return casos


# ----------------------------------------------------------------------------- página
def spec(chart: alt.Chart) -> str:
    return chart.configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12, titleColor="#444",
                                                            gridColor="#eceef1").to_json(indent=None)


def construir() -> None:
    df, ficha, resultados = cargar()
    dif = verificar_contra_el_pipeline(df)
    assert dif < 1e-9, f"La fórmula de la app no coincide con el modelo del notebook ({dif})"

    afuera = fuera_de_muestra(df, resultados)
    explorar = partidos_para_explorar(afuera)
    (SALIDA / "data").mkdir(parents=True, exist_ok=True)
    explorar.to_json(SALIDA / "data" / "partidos.json", orient="records", force_ascii=False)
    (SALIDA / ".nojekyll").write_text("", encoding="utf-8")

    g_datos, n_datos = grafico_datos(df)
    g_hallazgo, n_hallazgo = grafico_hallazgo(df)
    g_modelo = grafico_modelo(afuera)
    ligas = sorted(explorar["liga"].unique())
    temporadas = sorted(explorar["temporada"].unique())
    graficos = {"g_datos": spec(g_datos), "g_hallazgo": spec(g_hallazgo), "g_modelo": spec(g_modelo),
                "g_explorar": spec(grafico_explorar(ligas, temporadas)), "g_probar": spec(grafico_probar())}

    val, test = ficha["validacion"], ficha["test"]
    pct = lambda x, d=1: f"{100 * x:.{d}f}".replace(".", ",") + "%"  # noqa: E731
    num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    fecha_datos = json.loads(SILVER_MANIFEST_PATH.read_text(encoding="utf-8")).get("built_at_utc", "")[:10]

    rangos = {c: {"min": v["min_entrenamiento"], "max": v["max_entrenamiento"], "nombre": NOMBRES_GAPS[c]}
              for c, v in ficha["numericas"].items()}
    entrenamiento = df[df["temporada"] != TEST_TEMPORADA]
    rangos.update({c: {"min": float(entrenamiento[c].min()), "max": float(entrenamiento[c].max()), "nombre": c}
                   for c in ficha["probabilidades"]})
    contexto = {
        "TASA": pct(df[TARGET_COLUMN].mean()), "N": f"{len(df):,}".replace(",", "."),
        "N_EXPLORAR": f"{len(explorar):,}".replace(",", "."), "AUC_TEST_2": num(test["auc"], 2), "AUC_MERCADO_2": num(test["auc_mercado"], 2),
        "AUC_TEST": num(test["auc"]), "AUC_MERCADO": num(test["auc_mercado"]), "AUC_VAL": num(val["auc_media"]),
        "AUC_VAL_SD": num(val["auc_desvio"]), "ROI_TODOS": pct(val["roi_todos"]), "ROI_MODELO": pct(val["roi"]),
        "ROI_TEST": pct(test["roi"]), "ROI_TODOS_TEST": pct(test["roi_todos"]), "APUESTAS_TEST": str(test["apuestas"]),
        "DATOS_ALTO": pct(n_datos["alto"], 0), "DATOS_BAJO": pct(n_datos["bajo"], 0),
        "XI_BAJO": pct(n_hallazgo["bajo"], 0), "XI_ALTO": pct(n_hallazgo["alto"], 0),
        "FECHA_DATOS": fecha_datos, "EMBUDO": embudo(), "FUGA": ", ".join(f"<code>{c}</code>" for c in LEAKAGE_COLUMNS),
        "T_COMPARACION": tabla_html(resultados["comparacion"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}", "AUC train": "{:.3f}",
                                                                         "log-loss media": "{:.4f}", "Δ AUC % vs logística": "{:+.2f}%"}),
        "T_GRUPOS": tabla_html(resultados["grupos"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}",
                                                               "Δ AUC vs prob. sola": "{:+.4f}", "desvío de Δ": "{:.4f}"}),
        "T_PLATA": tabla_html(resultados["plata"], formatos={"% de partidos": "{:.1f}%", "ROI %": "{:+.2f}%"}),
        "T_PEDAZOS": tabla_html(resultados["plata_por_pedazo"], formatos={k: "{:+.1f}%" for k in resultados["plata_por_pedazo"][0] if k not in ("fila", "casa")}),
        "T_TEST": tabla_html(resultados["test"], formatos={"AUC": "{:.3f}", "log-loss": "{:.4f}", "ROI %": "{:+.2f}%"}),
        "T_COEF": tabla_html(resultados["coeficientes"], formatos={"coeficiente": "{:+.3f}"}),
        "C": str(ficha["C"]), "INTERCEPTO": num(ficha["intercepto"]),
        "COEF_PROB": num(ficha["probabilidades"]["prob_no_favorito"]["coef"]),
        "GRAFICOS": json.dumps(graficos, ensure_ascii=False),
        "MODELO": json.dumps({"intercepto": ficha["intercepto"], "probabilidades": ficha["probabilidades"],
                              "numericas": ficha["numericas"], "rangos": rangos}, ensure_ascii=False),
        "CASOS": json.dumps(casos_ejemplo(df), ensure_ascii=False),
        "VEGA": VEGA, "VEGA_LITE": VEGA_LITE, "VEGA_EMBED": VEGA_EMBED,
    }
    pagina = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
    for clave, valor in contexto.items():
        pagina = pagina.replace("{{" + clave + "}}", valor)
    assert "{{" not in pagina, "quedó un marcador sin reemplazar en la plantilla"
    (SALIDA / "index.html").write_text(pagina, encoding="utf-8")
    print(f"App generada en {SALIDA / 'index.html'} ({len(pagina) / 1024:.0f} KB) y "
          f"{SALIDA / 'data' / 'partidos.json'} ({len(explorar):,} partidos). Fórmula = notebook (dif. {dif:.1e}).")


if __name__ == "__main__":
    construir()
