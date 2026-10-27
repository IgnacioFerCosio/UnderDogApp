"""Transformaciones de columnas que usa el pipeline de modelado (Entrega 3).

Viven en el paquete, y no en el notebook, porque el modelo guardado con joblib
las referencia por nombre: la aplicación de la Entrega 4 tiene que poder
importarlas para cargar el modelo.
"""

from __future__ import annotations

import numpy as np

# Las probabilidades del Silver nunca son exactamente 0 o 1, pero se recorta
# igual para que el logit no devuelva infinito.
_EPS = 1e-6


def logit(p):
    """Probabilidad -> log-odds: log(p / (1 - p)). Es la escala en la que la
    regresión logística puede representar "creerle al mercado" (coef. 1)."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))
