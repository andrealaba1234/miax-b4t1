# Taller B4-T1 — Diseño de Redes Confiables (Justicia e Incertidumbre)

Clasificación neuronal para **concesión de crédito** (dataset *Home Credit Default
Risk*) con foco en que las decisiones sean **precisas**, **justas** (*Fair Learning*)
y **honestas** respecto a su incertidumbre.

Este repositorio cubre la **arquitectura customizada + modelo base** y, sobre todo,
el **Aprendizaje Justo (FAIR Loss)**: una función de coste que combina el error de
clasificación con una penalización por la **dependencia estadística** entre la
predicción del modelo y la **variable sensible** (género).

$$\mathcal{L}_{FAIR} = \text{BCE}(y, \hat{y}) \;+\; \lambda \cdot \big|\,\rho_{Pearson}(\hat{y},\, s)\,\big|$$

---

## 📁 Estructura del repositorio

```
miax-b4t1/
├── data/
│   ├── application_train.zip      # datos comprimidos (descomprimir antes de usar)
│   └── application_train.csv      # se obtiene al descomprimir (ignorado por git, ~159 MB)
├── notebooks/
│   ├── 01_base_model.ipynb        # capa customizada + modelo base (Parte 1)
│   └── 02_fair_loss.ipynb         # FAIR Loss: sweep de λ + Keras Tuner (Parte 2)
├── src/
│   └── base.py                    # FUENTE ÚNICA: loader + capa custom + modelo base
├── outputs/                       # entregables generados por 02_fair_loss.ipynb
│   ├── pareto_fairness.png        # Pareto Precisión vs Dependencia FAIR
│   ├── loss_curves.png            # curvas de convergencia Base vs mejor FAIR
│   └── tabla_base_vs_fair.csv     # tabla de resultados en test
├── results/
│   └── loss_curve_base.png        # curva de loss del modelo base (Parte 1)
├── docs/
│   └── Taller_B4_T1.pdf           # enunciado del taller
└── requirements.txt
```

> **`src/base.py` es la única fuente de verdad** del preprocesado, la capa
> customizada (*Ratio de Endeudamiento*) y la arquitectura. Ambos notebooks la
> importan, de modo que la comparación **Base vs FAIR** es *apples-to-apples*.

---

## ⚙️ Instalación

Requiere **Python 3.10**. Desde la raíz del repositorio:

```bash
# 1) (recomendado) crear un entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) instalar dependencias
pip install -r requirements.txt
```

Dependencias clave: `tensorflow==2.21.0`, `keras==3.12.2` (Keras 3),
`keras-tuner==1.4.8`, `tensorboard` (lo requiere keras-tuner), `scikit-learn`,
`scipy`, `pandas`, `numpy`, `matplotlib`.

---

## 📦 Preparación de los datos (¡importante!)

El CSV original (`application_train.csv`) pesa **~159 MB**, por encima del límite de
100 MB de GitHub, por lo que se distribuye **comprimido** en `data/application_train.zip`.
**Antes de ejecutar los notebooks hay que descomprimirlo** dentro de `data/`:

**Windows (PowerShell):**
```powershell
Expand-Archive -Path data\application_train.zip -DestinationPath data\ -Force
```

**Linux / Mac:**
```bash
unzip -o data/application_train.zip -d data/
```

Debe quedar el fichero `data/application_train.csv`. El loader lo busca en esa ruta
exacta.

---

## ▶️ Cómo ejecutar

Los notebooks detectan automáticamente la raíz del repositorio y se ejecutan con
**Run All** (semillas fijadas → reproducible). Ábrelos con Jupyter / VS Code:

```bash
jupyter notebook            # o: jupyter lab
```

1. **`notebooks/01_base_model.ipynb`** — EDA, capa customizada *Ratio de
   Endeudamiento* y entrenamiento del modelo base. Genera `results/loss_curve_base.png`.
2. **`notebooks/02_fair_loss.ipynb`** — la **FAIR Loss**: test de sanidad de la
   correlación, *sweep* manual de `λ`, búsqueda con **Keras Tuner**, Pareto y tabla
   final. Genera los entregables en `outputs/`.

> **Ejecución no interactiva** (regenerar todo desde terminal):
> ```bash
> jupyter nbconvert --to notebook --execute --inplace notebooks/01_base_model.ipynb
> jupyter nbconvert --to notebook --execute --inplace notebooks/02_fair_loss.ipynb
> ```
> El `02` tarda ~15 min en CPU (sweep de 8 valores de λ + 10 trials de Keras Tuner).

### Nota sobre el kernel de Jupyter
Asegúrate de ejecutar los notebooks con el **mismo entorno** donde instalaste
`requirements.txt` (selecciona ese kernel en Jupyter/VS Code). Si al importar
`keras_tuner` aparece `ModuleNotFoundError`, es que el kernel apunta a otro Python:
instala las dependencias en ese intérprete o cambia de kernel.

---

## 📊 Entregables (en `outputs/`)

| Fichero | Contenido |
|---|---|
| `pareto_fairness.png` | Curva de Pareto: **ROC-AUC** (Y) vs **dependencia FAIR** `\|corr(ŷ,s)\|` (X), un punto por `λ` del sweep + nube de trials del Keras Tuner. |
| `loss_curves.png` | Curvas de convergencia (train vs val) del modelo **Base** y del **mejor modelo FAIR**. |
| `tabla_base_vs_fair.csv` | Tabla en test: Accuracy, ROC-AUC, PR-AUC, `\|corr(ŷ,s)\|`, Demographic Parity Difference. |

### Resultados de referencia (test)

| Modelo | ROC-AUC | PR-AUC | \|corr(ŷ,s)\| | DPD |
|---|---|---|---|---|
| Base (λ=0) | 0.742 | 0.222 | 0.250 | 0.206 |
| **Mejor FAIR (λ=0.5)** | **0.733** | **0.212** | **0.019** | **0.015** |

Llevar la dependencia de género a ≈0 cuesta apenas **~1 punto de ROC-AUC**
(criterio de "mejor FAIR": máximo AUC con `|corr| ≤ 0.05`).

---

## 🔍 Detalles técnicos de la FAIR Loss

- **Correlación de Pearson diferenciable** implementada a mano con `keras.ops`
  (sin `scipy`/`numpy` dentro de la loss, para que el gradiente fluya), con
  `epsilon=1e-8` de estabilidad y **valor absoluto** (buscamos ≈0).
- La variable sensible se pasa a la loss empaquetando las etiquetas como
  `y = (N, 2)`: columna 0 = *target* (BCE), columna 1 = sensible (correlación).
- `class_weight` balanceado aplicado **dentro** de la loss (dataset ~8% positivos).
- Factory reutilizable `make_fair_loss(lambda_fair, base_error="bce")`
  (soporta `base_error="mse"`).
- El género (`CODE_GENDER`, **M=0, F=1**) se mantiene como *feature* de entrada y la
  FAIR loss se encarga de **suprimir su influencia** (no se usa *fairness through
  unawareness*).
