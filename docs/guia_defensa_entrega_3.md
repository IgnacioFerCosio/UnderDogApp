# GUÍA DE ESTUDIO Y DEFENSA ORAL — ENTREGA 3 · MODELADO
**Materia:** Ciencia de Datos · 5to Año · **Proyecto:** Winrate Underdog 5K10
**Defensa:** miércoles 28/10 · 25 minutos en privado con el docente · **la nota es INDIVIDUAL**

Notebook: `ENtrega 2.0/UnderDog/notebooks/entrega_3_modelado.ipynb` (ejecutado, con salidas guardadas, semilla 42).
**Ábranlo ANTES de entrar.** No hace falta correr nada en vivo: todas las salidas ya están guardadas.

---

## 0. LA IDEA EN 30 SEGUNDOS (todos tienen que poder decir esto)

> "Predecimos si gana el underdog, que gana el 20% de las veces. Medimos con ROC-AUC porque lo que queremos es **ordenar**
> los partidos por chance de sorpresa, y la accuracy engaña: decir siempre 'no gana' da 80%. Validamos **en el tiempo**,
> entrenando con el pasado y validando con lo que viene, porque así se va a usar. Comparamos logística, árbol, random forest y
> boosting, y **ninguno le ganó a la logística**. Lo más fuerte: los atributos de los jugadores **no le agregan nada al
> mercado**. El modelo final aprende `logit(p) = −0,04 + 1,00 · logit(prob. del mercado)`, o sea que **la probabilidad de las
> casas de apuestas ya está calibrada**. Le ganamos al piso por mucho (AUC 0,66 contra 0,5) y empatamos con el mercado."

---

## 1. NÚMEROS DE MEMORIA

| Qué | Valor | Para qué lo usás |
|---|---|---|
| Dataset | **17.959 partidos × 31 columnas** (igual que E2) | "no cambió nada del dataset" |
| Target | `gano_no_favorito`: **20,35%** gana / 79,65% no | por qué no accuracy |
| Entrenamiento | 2008/09 a 2014/15 · **15.560 partidos** · 20,1% gana | |
| Test | **temporada 2015/16** · **2.399 partidos** · 22,1% gana | se mira UNA vez |
| Validación | `TimeSeriesSplit(n_splits=5)`: 5 bloques de 2.593 partidos | |
| Piso (Dummy) | AUC **0,500** · log-loss **0,502** (val) / **0,529** (test) · accuracy **0,80** | |
| Cuota cruda del mercado | AUC **0,637** (val) / **0,656** (test) · log-loss 0,481 / 0,498 | el rival real |
| Logística 12 columnas | AUC **0,636 ± 0,009** | referencia |
| Random forest | 0,632 ± 0,013 (train 0,676: sobreajusta) | |
| Gradient boosting | 0,632 ± 0,012 | |
| Árbol de decisión | 0,624 ± 0,011 (sin límite: train 1,000 / val 0,52) | curva de complejidad |
| Solo jugadores (10 col.) | AUC **0,599** | los jugadores solos sí ordenan |
| Mercado + jugadores vs mercado | Δ AUC **−0,0002 ± 0,003** | no suman nada |
| **MODELO FINAL** | Logística, C=1, **1 columna**: logit(prob. del underdog) | |
| Ecuación aprendida | **logit(p) = −0,039 + 1,004 · logit(prob. mercado)** | EL hallazgo |
| Final en validación | AUC **0,637 ± 0,010** · log-loss 0,481 | |
| Final en test | AUC **0,656** · log-loss **0,499** | |
| Umbral elegido | **0,172** (máximo f1 de la clase "gana", en validación) | |
| Con ese umbral (test) | precisión 0,29 · recall 0,85 · f1 0,43 | |
| Al azar vs temporal | −0,0025 (logística): no infla | chequeo de fuga |

---

## 2. GUION DE LOS 25 MINUTOS

Repártanse las partes, pero **TODOS estudian TODO**: la consigna dice que la pregunta sobre validación le puede caer a quien
armó las columnas.

### Min 0–2 · Qué cambió (dos oraciones)
*"El objetivo y el dataset son los mismos de la Entrega 2: 17.959 partidos, target `gano_no_favorito`, las 12 columnas en
amarillo/verde. Lo que corregimos es la lectura: todos los gráficos tienen título, ejes en castellano y nombres legibles, que era
lo que se veía poco claro en la Entrega 2."*

### Min 2–7 · El planteo (Parte 1 y 2 del notebook)
1. **Qué predecimos**: si gana el underdog. Clasificación binaria, 20/80.
2. **Métrica**: ROC-AUC principal, log-loss secundaria. Mostrar la tabla del piso: **accuracy 0,80 sin mirar nada**.
3. **Partición**: test = última temporada completa (2015/16). Mostrar la tabla de los 5 pedazos de `TimeSeriesSplit`.
4. **Pipeline**: logit de las probabilidades + imputar + escalar + modelo, todo en un objeto (`armar()`).
5. **Piso y rival**: Dummy 0,5; cuota cruda 0,637. *"El piso es fácil de superar; el rival de verdad es el mercado."*

### Min 7–16 · El resultado (Partes 3 a 7)
1. **Gráfico de comparación** (Parte 3.1): todos los modelos en 0,62–0,64, con barras que se pisan.
2. **Tabla de la regla** (3.2): ningún complejo gana por más que el desvío, y en la diferencia pedazo a pedazo ganan 0–2 de 5.
3. **Curva del árbol** (3.3): train sube hasta 1,0 y validación cae a 0,52. Eso es sobreajuste.
4. **Al azar vs temporal** (3.4): no cambia, y explicar por qué (ver pregunta 3).
5. **Columnas** (Parte 5): solo jugadores 0,60; sumados al mercado no aportan; el logit sí mejora la log-loss en los 5 pedazos.
6. **Desbalance** (Parte 6): `class_weight` infla las probabilidades (media 0,48 cuando la real es 0,20) → no lo usamos;
   movimos el umbral a 0,172.
7. **Test una vez** (Parte 7): 0,656. Matrices de confusión, calibración, tramos, errores concretos.
8. **En qué se apoya**: coeficiente 1,004 → el modelo es "creerle al mercado".

### Min 16–22 · Preguntas dirigidas → sección 3 de esta guía
### Min 22–25 · Devolución: anotar todo para la Entrega 4

---

## 3. LAS PREGUNTAS DE LA CONSIGNA, CON RESPUESTA

### P1. ¿Por qué esa métrica? ¿Qué error es más caro y la métrica lo ve?
- **Accuracy no**: el Dummy da 0,80 sin mirar nada y esconde a la clase chica, que es la que nos importa.
- **El error caro es ordenar mal**: la app tiene que decir qué partidos tienen más chance de sorpresa. ROC-AUC mide justo eso:
  *la probabilidad de que, tomando un partido que el underdog ganó y otro que no, el modelo le dé más chance al que ganó*.
  No depende del umbral ni del desbalance, y el piso vale 0,5 exacto.
- **Log-loss** porque la app **muestra** una probabilidad: si decimos 30%, tiene que ganar ~30% de esas veces. Un modelo puede
  ordenar bien y dar probabilidades mentirosas (ejemplo: `class_weight`, misma AUC y log-loss mucho peor).
- ¿Por qué no f1 macro? El f1 depende del umbral. Lo usamos **después**, para elegir el umbral del sí/no.

### P2. ¿Cuánto le gana al piso? ¿Es mucho o poco?
- **AUC 0,656 contra 0,5** en test (+0,16); en validación +0,137 con desvío 0,01: **más de 10 desvíos**. Log-loss 0,499 contra 0,529.
- **¿Es mucho?** Contra el piso, sí. Contra el mercado, **cero**: empatamos (0,656 contra 0,656). Para la app alcanza para
  ordenar partidos (del 3% al 31% de victorias entre tramos), pero no para "ganarle a las casas de apuestas".
- Ojo con la trampa: *"¿0,656 es bueno?"*. Sí, para fútbol. Es un deporte de pocos goles y mucho azar: un penal o una expulsión
  decide. En nuestra base, ni siquiera el mercado, con toda su información, pasa de 0,64–0,66.

### P3. ¿Por qué los pedazos se arman así? ¿Qué pasaría al azar?
- **Así se usa el modelo**: se entrena con lo jugado y se predice lo que viene. `TimeSeriesSplit` entrena con el pasado y
  valida con el bloque siguiente. El test es la temporada siguiente entera.
- **Al azar se mezclaría el futuro en el entrenamiento**: entrenar con partidos de 2015 para predecir 2010. Los ratings FIFA
  cambian de temporada a temporada, y la casa de apuestas también: Bet365 hasta 2011/12 y Pinnacle desde 2012/13.
- **Lo medimos y NO se infló** (−0,0025). ¿Por qué? Porque **ninguna columna identifica al equipo ni a la fecha**: son
  diferencias de rating y probabilidades. Las que sí identificarían (`equipo_local`, `equipo_visitante`, `fecha`) quedaron
  afuera de Silver en la Entrega 2. Si alguna columna "supiera" algo del partido o del equipo, el azar la hubiera premiado.
  **Este resultado es además un chequeo de fuga.**
- Si preguntan por `GroupKFold`: los equipos se repiten, pero como no hay columna de equipo, el modelo no puede memorizarlos.
  El corte temporal ya garantiza que ningún partido quede de los dos lados.
- Detalle de la tabla: "entrena hasta" y "valida desde" pueden caer el mismo día, porque `TimeSeriesSplit` corta por fila. Son
  partidos distintos jugados a la vez: ninguno ve el resultado del otro.

### P4. ¿Por qué ese modelo y no uno más complejo?
- **Regla de la consigna**: si el complejo no le gana al simple por más que el desvío, se queda el simple.
- Random forest −0,004, boosting −0,004 y árbol −0,012, con desvíos de 0,011 a 0,013. Ninguno gana; pedazo a pedazo ganan en 0 a 2 de 5.
- **Cada familia tuvo su grilla** (no es un "zoológico" de valores por defecto): árbol 12 combinaciones, RF 6, boosting 8, logística 5.
- **Por qué el complejo no gana**: la señal es casi lineal en la cuota. Un árbol o un bosque solo pueden encontrar
  interacciones que no existen, y terminan memorizando. RF: train 0,676 contra validación 0,632. Árbol sin límite: 1,0 contra 0,52.
- La misma regla aplicada a **las columnas**: 1 columna → 2 (+empate) → 12 (+jugadores), y ninguna escalera mejora más que su
  desvío. Por eso el final tiene **una sola columna**.

### P5. Una confusión concreta y su explicación desde el dominio
- **Sorpresas que no vio**: Bayern 1–2 Mainz (03/2016), Barcelona 1–2 Valencia (04/2016), Leverkusen 0–1 Darmstadt (recién
  ascendido). El modelo les daba 4–7% a los underdogs, igual que el mercado. **No es un error de lectura**: con 1 en 20, uno de
  cada 20 de esos partidos lo gana el chico. Lo decide un gol que ninguna columna pre-partido puede saber.
- **Falsas alarmas**: partidos con el underdog en ~33% (Werder–Wolfsburg, Granada–Sevilla). Aun con la máxima chance, 2 de cada 3 veces
  el underdog no gana. Es la incertidumbre del partido.
- **Por tramo**: adentro de cada tramo de cuotas, la AUC es ~0,5. El modelo sabe *qué tan parejo* es el partido, pero entre dos
  partidos igual de parejos no sabe cuál va a ser la sorpresa.
- **Calibración en el test**: en los deciles del medio el underdog ganó 27% contra 20–23% predicho, y **el mercado erró igual**.
  2015/16 fue una temporada más sorpresiva (22,1% contra 20,1%): la del Leicester campeón. Ningún modelo entrenado con el pasado
  podía saberlo.

### P6. La columna que más pesa: ¿cómo descartaron fuga?
La que más pesa es la **probabilidad implícita del underdog**: si se desordena, la AUC cae de 0,656 a ~0,50.
1. **Existe antes del partido**: sale de las cuotas 1X2 publicadas días antes. Los goles y el resultado nunca entran a Silver,
   y el DAG falla si entran (hard check `sin_columnas_de_fuga`).
2. **No predice "demasiado bien"**: sola da AUC 0,656, lejos del 0,90 a partir del cual el DAG falla por predictor perfecto. Con
   la respuesta adentro daría ~1.
3. **No hay brechas**: train contra validación −0,007, validación contra test −0,02 (el test da *mejor*), azar contra temporal −0,0025.
4. **El coeficiente es ~1**: el modelo no la usa "de más", solo la confirma.
5. **Tiene sentido de dominio**: es la opinión agregada del mercado (forma, lesiones, localía, plantel).

---

## 4. PREGUNTAS TRAMPA PROBABLES

**"Si el modelo es la cuota, ¿para qué el modelo?"**
→ Para **saberlo con evidencia**. Antes de modelar no sabíamos si el mercado estaba calibrado ni si los jugadores aportaban.
Ahora lo medimos: coeficiente 1,00, intercepto ~0 y jugadores con Δ 0,000. *"¿Tenés un modelo que dice algo que los datos solos no
decían?"* Sí: **que no hay información pública pre-partido, de las que tenemos, que mejore a las cuotas.** La consigna dice que
un resultado que no anduvo, bien entendido, vale. Este anduvo contra el piso y se explica contra el mercado.

**"¿Por qué el logit?"**
→ La logística modela `logit(p) = a + b·x`. Si `x = logit(prob. mercado)`, "creerle al mercado" es a=0, b=1, y el modelo lo
puede representar. Con la probabilidad cruda no puede (una sigmoide de algo lineal en `prob` no es una recta). Lo medimos: el
logit mejora la log-loss en los 5 pedazos. La AUC es la misma, porque el orden no cambia. Es una función fija, no aprende nada, y
vive en `include/src/features.py` adentro del pipeline.

**"¿Por qué el coeficiente de 'Dif. rating de los 3 mejores' es negativo?"**
→ **Colinealidad**. Medio, top 3 y mejor jugador correlacionan 0,86–0,96 (E2). La logística reparte el peso entre columnas que
dicen lo mismo y los signos sueltos no se interpretan. Es otra razón para no usar las 12.

**"¿Por qué no `class_weight='balanced'`?"**
→ La AUC queda igual (0,637), pero la log-loss pasa de 0,481 a 0,658: la probabilidad media predicha sube a 0,48 cuando la
tasa real es 0,20. Para una app que **muestra** la probabilidad, es mentirle al usuario. Movimos el umbral en su lugar.

**"Con el umbral 0,172 marcan 'gana' en 60% de los partidos. ¿Eso sirve?"**
→ Es la parte débil y lo decimos. El máximo f1 termina en una regla simple: "partido parejo → puede ganar el underdog". Precisión
0,29 contra 0,22 de base. **Lo valioso para la app es la probabilidad, no el sí/no.**

**"¿Por qué el test da mejor que la validación (0,656 contra 0,637)?"**
→ Está dentro de 2 desvíos: los 5 pedazos de validación ya varían entre sí ±0,01. Además, la cuota cruda **también** da 0,656
en ese test, así que no es algo del modelo: es la temporada. Una hipótesis (no medida) es que casi todo 2015/16 usa Pinnacle,
la casa más eficiente. Si diera *mucho* mejor, sospecharíamos. Y no tocamos nada después de mirarlo.

**"¿Miraron el test más de una vez?"**
→ No. Todo se eligió con validación cruzada en el entrenamiento: familia, C, columnas, class_weight y umbral. El test aparece
recién en la Parte 7. Piso y mercado van en la misma tabla como referencia, pero no se elige nada con ellos.

**"¿Por qué no probaron con las 12 columnas igual, si no cuesta nada?"**
→ Cuesta interpretabilidad (signos sin sentido por colinealidad), cuesta para la app (12 entradas, con alineaciones que se conocen
1 h antes contra 1 cuota) y no gana nada: Δ −0,0002 ± 0,003. La regla es la misma que para los modelos.

**"¿Qué hubieran necesitado para ganarle al mercado?"**
→ Información que el mercado no tenga o incorpore tarde: **cuotas de cierre** contra las de apertura (en la base no hay), forma
reciente, lesiones y suspensiones (tampoco hay). Las alineaciones salen 1 h antes y las cuotas días antes, y aun así no
sumaron: el mercado ya "sabe" quién es mejor.

**"¿De dónde sale el Silver que usa el notebook?"**
→ **Del DAG**, igual que en la E2: el `database.sqlite` se carga **una vez** a PostgreSQL (`source-db`) con el seed, y el
DAG consulta esa base por la Airflow Connection `soccer_source_db` (las consultas de `include/sql/`) → Bronze → Silver.
El notebook solo lee `include/data/silver/underdog_dataset.parquet`. Además verificamos que es **idéntico** al que da correr
las mismas consultas contra el SQLite (`scripts/build_silver_local.py`, que es solo una herramienta de verificación).

**"¿Y si corren el DAG de nuevo, recarga todo?"**
→ No. El seed compara las filas de cada tabla y si ya están, las saltea (la segunda corrida tarda 8 segundos). El DAG calcula
una huella de la fuente y del código; si no cambió nada desde la última corrida exitosa, el `ShortCircuit` saltea el resto.

---

## 5. GLOSARIO EXPRESS (definiciones de una línea)

- **Pipeline**: el objeto que encadena preparar + modelo. Se ajusta solo con el train de cada pedazo, así nada "ve" la validación.
- **Validación cruzada**: entrenar y medir varias veces con pedazos distintos; se reporta media ± desvío.
- **TimeSeriesSplit**: validación cruzada donde siempre se entrena con el pasado y se valida con el bloque siguiente.
- **ROC-AUC**: probabilidad de que el modelo le dé más chance a un caso positivo que a uno negativo, ambos al azar. 0,5 = azar, 1 = perfecto.
- **Log-loss**: castiga las probabilidades confiadas y equivocadas. Más bajo es mejor. Mide calibración + orden.
- **Calibración**: que el "30%" del modelo signifique 30% de victorias reales.
- **Sobreajuste**: train alto y validación baja. El modelo memoriza en vez de aprender (el árbol sin límite).
- **Regularización (C)**: C chico frena los coeficientes. Acá casi no cambia nada desde C=0,1.
- **class_weight**: pesar más los errores de la clase chica. Cambia las probabilidades, no el orden.
- **Umbral**: corte sobre la probabilidad para decir sí/no. Se eligió con predicciones fuera de muestra de la validación.
- **Importancia por permutación**: desordenar una columna y ver cuánto cae la métrica.
- **Fuga (leakage)**: que entre al modelo información que no existe al momento de predecir.
- **logit**: log(p / (1−p)), la escala natural de la regresión logística.
- **Underdog**: el de menor probabilidad implícita (con gap > 0,05 con el favorito).

---

## 6. CHECKLIST ANTES DE ENTRAR

- [ ] Notebook abierto en `notebooks/entrega_3_modelado.ipynb`, scrolleado arriba.
- [ ] Cada integrante puede decir **la idea en 30 segundos** (sección 0).
- [ ] Cada integrante puede responder **P3 (validación)** y **P6 (fuga)**: son las más caras.
- [ ] Saber dónde está en el notebook: la tabla del piso (P2), el gráfico de comparación (P4), los errores concretos (P5) y el coeficiente 1,004.
- [ ] Nadie habla solo más de 5 minutos seguidos.
- [ ] Tener a mano "lo que probamos y no anduvo" (tabla de la Parte 8): la consigna lo pide explícitamente.
