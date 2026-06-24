"""Módulo base compartido del Taller B4-T1 (Diseño de Redes Confiables).

Este módulo es la ÚNICA fuente de verdad para:
  1. La carga y preprocesado del dataset Home Credit (`load_home_credit_data`).
  2. La capa customizada de "Ratio de Endeudamiento" (`RatioEndeudamientoLayer`).
  3. El constructor del modelo base (`build_model`).

Tanto `01_base_model.ipynb` (modelo Base) como `02_fair_loss.ipynb` (FAIR loss)
importan de aquí, de modo que la comparación Base vs FAIR usa exactamente la misma
arquitectura y el mismo preprocesado (comparación apples-to-apples).

Solo Keras 3 / tf.keras / keras.ops. Sin APIs depreciadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras import ops


# ---------------------------------------------------------------------------
# Orden de features que produce el loader (depende del orden del CSV).
# IMPORTANTE: la variable sensible CODE_GENDER queda DENTRO de X (índice 0).
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "CODE_GENDER",        # 0  <- variable sensible (M=0, F=1), presente en la entrada
    "AMT_INCOME_TOTAL",   # 1
    "AMT_CREDIT",         # 2
    "AMT_ANNUITY",        # 3
    "DAYS_BIRTH",         # 4
    "EXT_SOURCE_1",       # 5
    "EXT_SOURCE_2",       # 6
    "EXT_SOURCE_3",       # 7
    "EXT_SOURCE_1_NULL",  # 8
    "EXT_SOURCE_2_NULL",  # 9
    "EXT_SOURCE_3_NULL",  # 10
]

# Índices financieros usados por la capa de Ratio de Endeudamiento.
IDX_INCOME = FEATURE_NAMES.index("AMT_INCOME_TOTAL")   # 1
IDX_CREDIT = FEATURE_NAMES.index("AMT_CREDIT")         # 2
IDX_ANNUITY = FEATURE_NAMES.index("AMT_ANNUITY")       # 3
IDX_GENDER = FEATURE_NAMES.index("CODE_GENDER")        # 0


# ===========================================================================
# 1. Carga y preprocesado (idéntico al 01_base_model.ipynb original)
# ===========================================================================
def load_home_credit_data(file_path):
    """Carga, limpia y divide el dataset Home Credit.

    Devuelve `(X_train, y_train, s_train), (X_test, y_test, s_test)` con split
    estratificado 80/20 (random_state=42) y `StandardScaler` ajustado en train.
    `s` = variable sensible (género: M=0, F=1).
    """
    cols = [
        "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]

    df = pd.read_csv(file_path, usecols=cols)

    # Eliminar filas con género desconocido
    df = df[df["CODE_GENDER"].isin(["M", "F"])]

    # Género a numérico (variable sensible): M=0, F=1
    df["CODE_GENDER"] = df["CODE_GENDER"].map({"M": 0, "F": 1})

    # Edad a años positivos
    df["DAYS_BIRTH"] = abs(df["DAYS_BIRTH"]) / 365

    # Imputar nulos de EXT_SOURCE con la mediana y crear flags _NULL
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df[col + "_NULL"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median())

    # Imputar AMT_ANNUITY
    df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].median())

    # Separar X, y, s
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    s = df["CODE_GENDER"]

    # Split train/test estratificado
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado (ajustado solo en train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train, s_train), (X_test, y_test, s_test)


# ===========================================================================
# 2. Capa customizada: Ratio de Endeudamiento (restricción matemática)
# ===========================================================================
@keras.saving.register_keras_serializable(package="b4t1")
class RatioEndeudamientoLayer(keras.layers.Layer):
    """Capa customizada que inyecta una señal de "carga de endeudamiento".

    Calcula internamente una medida de endeudamiento combinando variables
    financieras de entrada y le aplica una **saturación tanh** (restricción
    matemática) antes de concatenarla al vector de features original.

    Razonamiento de la señal (sobre features YA estandarizadas):
      carga = w_credit * z(AMT_CREDIT) + w_annuity * z(AMT_ANNUITY)
              - w_income * z(AMT_INCOME_TOTAL)
    Más crédito/anualidad respecto a los ingresos => mayor carga.
    La saturación `tanh` acota la señal a (-1, 1), evitando que perfiles con
    apalancamiento extremo dominen las capas densas (restricción de saturación).

    La capa NO tiene parámetros entrenables: es una transformación determinista
    (una "restricción física/matemática" en el sentido del enunciado). La salida
    tiene dimensión `input_dim + 1` (se añade la señal de endeudamiento).
    """

    def __init__(self, idx_income=IDX_INCOME, idx_credit=IDX_CREDIT,
                 idx_annuity=IDX_ANNUITY, w_income=1.0, w_credit=1.0,
                 w_annuity=1.0, **kwargs):
        super().__init__(**kwargs)
        self.idx_income = idx_income
        self.idx_credit = idx_credit
        self.idx_annuity = idx_annuity
        self.w_income = w_income
        self.w_credit = w_credit
        self.w_annuity = w_annuity

    def call(self, inputs):
        # Selección de columnas financieras (estandarizadas)
        income = inputs[:, self.idx_income]
        credit = inputs[:, self.idx_credit]
        annuity = inputs[:, self.idx_annuity]

        # Señal de carga de endeudamiento
        carga = (self.w_credit * credit
                 + self.w_annuity * annuity
                 - self.w_income * income)

        # Restricción matemática: saturación tanh -> (-1, 1)
        endeudamiento = ops.tanh(carga)
        endeudamiento = ops.expand_dims(endeudamiento, axis=-1)  # (batch, 1)

        # Concatenar la señal acotada al vector de entrada original
        return ops.concatenate([inputs, endeudamiento], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] + 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "idx_income": self.idx_income,
            "idx_credit": self.idx_credit,
            "idx_annuity": self.idx_annuity,
            "w_income": self.w_income,
            "w_credit": self.w_credit,
            "w_annuity": self.w_annuity,
        })
        return config


# ===========================================================================
# 3. Constructores de modelos (arquitectura compartida Base/FAIR)
# ===========================================================================
def build_model_from_config(
    input_dim,
    hidden_units,
    dropouts=None,
    activation="relu",
    use_custom_layer=True,
    output_activation="sigmoid",
    output_name="pd",
    name="credit_model",
):
    """Construye (sin compilar) un MLP con capa customizada y topología variable.

    Parámetros
    ----------
    input_dim : int
        Número de features de entrada.
    hidden_units : list[int] | tuple[int, ...]
        Unidades de cada capa densa oculta.
    dropouts : list[float] | tuple[float, ...] | None
        Dropout aplicado tras cada capa oculta. Si es `None`, se usa 0.0
        en todas. Si se pasa un escalar, se replica para todas las capas.
    activation : str
        Activación de las capas ocultas.
    use_custom_layer : bool
        Si `True`, inserta `RatioEndeudamientoLayer` justo tras la entrada.
    output_activation : str
        Activación de la capa de salida.
    output_name : str
        Nombre de la capa de salida.
    name : str
        Nombre del modelo Keras.
    """
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)
    if len(hidden_units) == 0:
        raise ValueError("hidden_units debe contener al menos una capa oculta.")

    if dropouts is None:
        dropouts = [0.0] * len(hidden_units)
    elif isinstance(dropouts, (int, float)):
        dropouts = [float(dropouts)] * len(hidden_units)
    else:
        dropouts = list(dropouts)

    if len(dropouts) != len(hidden_units):
        raise ValueError("dropouts debe tener la misma longitud que hidden_units.")

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    if use_custom_layer:
        x = RatioEndeudamientoLayer(name="ratio_endeudamiento")(x)

    for i, (units, dropout) in enumerate(zip(hidden_units, dropouts), start=1):
        x = keras.layers.Dense(units, activation=activation, name=f"dense_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = keras.layers.Dense(
        1, activation=output_activation, name=output_name
    )(x)
    return keras.Model(inputs, outputs, name=name)


def build_model(input_dim, units1=64, units2=32, dropout=0.2,
                use_custom_layer=True):
    """Construye (sin compilar) el modelo base + capa customizada.

    Arquitectura:
        Input(input_dim)
          -> RatioEndeudamientoLayer  (capa customizada, opcional)
          -> Dense(units1, relu) -> Dropout
          -> Dense(units2, relu) -> Dropout
          -> Dense(1, sigmoid)

    Se devuelve SIN compilar para que cada notebook lo compile con su loss:
    el 01 con BCE estándar y el 02 con la FAIR loss. Así la arquitectura es
    idéntica y la comparación es justa.
    """
    return build_model_from_config(
        input_dim=input_dim,
        hidden_units=[units1, units2],
        dropouts=[dropout, dropout],
        activation="relu",
        use_custom_layer=use_custom_layer,
        output_activation="sigmoid",
        output_name="pd",
        name="credit_model",
    )


def compute_class_weight_balanced(y):
    """Class weight balanceado (dataset ~8% positivos)."""
    y = np.asarray(y).ravel()
    n = len(y)
    n_pos = y.sum()
    n_neg = n - n_pos
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}
