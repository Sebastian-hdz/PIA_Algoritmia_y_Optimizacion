# Hiper-heurística de Selección para TSP (Proyecto Final AYO)

Dada una instancia de TSP, el sistema predice cuál de tres algoritmos obtiene el mejor tour:

- `2opt_hill_climbing`: búsqueda local 2-opt (mejora primera, inicio aleatorio)
- `sa_2opt`: recocido simulado con vecindad 2-opt
- `mst_2opt`: aproximación MST (Prim + preorden) refinada con 2-opt

El proyecto tiene dos partes: (1) correr los tres algoritmos sobre cada instancia para generar las
etiquetas `best_algorithm`, y (2) entrenar y evaluar un clasificador (Árbol de Decisión y KNN).

## Equipo

| # | Rol | Archivo(s) | Nombre de los integrantes |
|---|-----|-----------| -------------------------- |
| 1 | Algoritmos | `tsp_algorithms.py` | Sthefany Paola Díaz Pérez |
| 2 | Etiquetado | `label_instances.py` | Elena Guadalupe Perez Contreras |
| 3 | Modelos | `train_eval.py` (`train_models`) | Jaime Sebastián Gonzalez Ontiveros |
| 4 | Métricas | `train_eval.py` (`evaluate_models`) | Mauricio López García |
| 5 | Análisis y repo | `train_eval.py` (`analyze_results`), `README.md` | Sebastián Hernández Pérez |



## Estructura

```
.
├── tsp_algorithms.py        # los 3 algoritmos
├── label_instances.py       # genera train_labeled.csv / test_labeled.csv
├── train_eval.py            # entrenamiento (parte 1), métricas en test (parte 2), análisis (parte 3)
├── train_labeled.csv        # train etiquetado (351 filas)
├── test_labeled.csv         # test etiquetado (149 filas)
├── labeling_summary.txt     # parámetros de SA, regla de empates, balance de clases, baseline
├── hh_tsp_dataset/          # dataset del profesor (instancias y características, sin etiquetas)
├── results/                 # modelos (.joblib), tablas (.csv) y figuras (.png)
├── requirements.txt
└── README.md
```

## Requisitos

Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correr

Todos los comandos se ejecutan desde la raíz del repositorio.

**1. Generar las etiquetas** (opcional: los CSV etiquetados ya vienen en el repo)

```bash
python label_instances.py --data hh_tsp_dataset/data --out .
```

Es determinista (semilla base 42) y tarda menos de un minuto. Con `--workers N` corre en paralelo
y el resultado no cambia. Escribe `train_labeled.csv`, `test_labeled.csv` y `labeling_summary.txt`.

**2. Entrenar, evaluar y analizar**

```bash
python train_eval.py                  # las tres partes en orden
python train_eval.py --only-train     # solo entrenamiento + validación cruzada (en train)
python train_eval.py --only-eval      # solo métricas en test (usa los modelos guardados)
python train_eval.py --only-analysis  # solo análisis (usa los modelos guardados)
```

El test solo se usa en las partes 2 y 3. Los hiperparámetros se eligen con validación cruzada
estratificada de 5 pliegues sobre train.

## Parámetros del etiquetado

| Parámetro | Valor |
|-----------|-------|
| SA: temperatura inicial T0 | 0.1 |
| SA: temperatura final | 0.0001 |
| SA: enfriamiento | exponencial, T_k = T0 · (T_final / T0)^(k/N) |
| SA: iteraciones | 30000 |
| Semilla base | 42 |
| Regla de empates | 2opt_hill_climbing > sa_2opt > mst_2opt (costos redondeados a 9 decimales) |

## Balance de clases (500 instancias, etiquetas propias)

| Algoritmo ganador | Instancias | % | Train | Test |
|-------------------|-----------:|--:|------:|-----:|
| 2opt_hill_climbing | 246 | 49.2% | 172 | 74 |
| sa_2opt | 182 | 36.4% | 128 | 54 |
| mst_2opt | 72 | 14.4% | 51 | 21 |

Baseline trivial («siempre 2opt_hill_climbing»): 49.7% de accuracy en test.
El 46.2% de las instancias tiene empate en el costo mínimo (175 con empate de los 3), resuelto por la regla de prioridad.

## Resultados

Modelo elegido por validación cruzada: **KNN** (k=15, p=1, pesos por distancia), accuracy CV 0.701 contra baseline CV 0.490.
Árbol: profundidad 2, criterio entropy, accuracy CV 0.667.

Resultados en test (149 instancias):

| Política / modelo | Accuracy | Accuracy con empates | Regret relativo medio |
|-------------------|---------:|---------------------:|----------------------:|
| Baseline (siempre 2-opt) | 49.7% | 49.7% | 2.19% |
| Árbol de decisión | 66.4% | 67.1% | 1.25% |
| KNN | 62.4% | 68.5% | 1.68% |

La comparación contra «siempre SA+2-opt» (regret relativo 1.36%) y el análisis completo están en el reporte.

### Hallazgos del análisis

- El tamaño n es la característica más importante en ambos modelos.
- 2-opt gana en instancias pequeñas (en buena parte por empates), SA entre n = 20 y n = 80, y MST+2-opt solo es competitivo con n grande.
- El árbol nunca predice `mst_2opt`; el KNN acierta 7 de las 21 instancias de esa clase.
- Con los datos de test no se puede afirmar que los modelos mejoren a la política fija «siempre SA+2-opt».

### Salidas del análisis (`results/`)

| Archivo | Contenido |
|---------|-----------|
| `feature_importance.csv`, `.png` | Importancia por impureza (árbol) y por permutación en test (árbol y KNN) |
| `permutation_groups.csv` | Permutación por grupos de características |
| `feature_correlation.csv` | Correlación de Spearman entre características numéricas |
| `patterns_by_distribution.csv` | Ganador por distribución |
| `patterns_by_size.csv` | Ganador por tamaño n |
| `patterns_by_dist_size.csv` | Ganador por (distribución, n) |
| `patterns_strict_by_dist_size.csv` | Igual, solo victorias sin empate |
| `relative_gap_by_size.csv` | Brecha relativa media de cada algoritmo contra el mejor, por n |
| `patterns_heatmaps.png`, `winners_by_size.png` | Figuras de patrones |
| `class_recall_test.csv` | Precision y recall por clase en test, y detección de «SA o MST superan a 2-opt» |
| `analysis_summary.txt` | Resumen en texto de todo el análisis |

Las salidas de la parte 2 (`test_metrics.*`, `confusion_*`, `regret_per_instance.csv`) también están en `results/`.

## Reproducibilidad

Semilla fija (42) en etiquetado, validación cruzada y permutaciones. Los modelos guardados en `results/` se entrenaron
con scikit-learn 1.8.0; con otra versión puede aparecer una advertencia al cargarlos (los resultados no cambian), o
se pueden reentrenar con `python train_eval.py --only-train`.
