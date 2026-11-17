"""
Proyecto Final - Hiper-heurística de Selección para TSP

QUÉ HACE
    Para cada instance_id de train.csv y test.csv: carga las coordenadas, corre los tres algoritmos
    de la Tarea 2 (2-opt hill climbing, SA + 2-opt, MST + 2-opt), registra el costo del tour que obtiene
    cada uno y agrega cuatro columnas:
        cost_2opt_hill_climbing, cost_sa_2opt, cost_mst_2opt, best_algorithm (el de menor costo)

USO
    python label_instances.py                          # espera hh_tsp_dataset/data/ en la carpeta actual
    python label_instances.py --data RUTA/A/data       # otra ubicación del dataset
    python label_instances.py --workers 4              # en paralelo (el resultado NO cambia)

ENTRADAS   <data>/train.csv, <data>/test.csv, <data>/instances/<instance_id>.csv
SALIDAS    <out>/train_labeled.csv, <out>/test_labeled.csv   (<out> = <data> por defecto)
           <out>/labeling_summary.txt  (parámetros, balance de clases, empates, baseline)

Requiere: pandas y el módulo tsp_algorithms.py (en la misma carpeta que este script).
"""
import argparse
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import tsp_algorithms as alg                 # módulo 1 (ver ADAPTADORES más abajo)

# =============================================================================
# PARÁMETROS DEL ETIQUETADO  (se documentan en el reporte)
# =============================================================================
# [REQUISITO: «Define los parámetros de SA»  y  «Documenta en tu reporte qué parámetros de SA usaste»]
# Son los valores del experimento de la Tarea 2 (Ej. 4: esquema exponencial, T0 = 10, T_final = 0.01, 30,000
# iteraciones, con coordenadas en [0,100]^2) convertidos a las coordenadas [0,1]^2 de este proyecto:
# se multiplican T0 y T_final por 1/100. La probabilidad de aceptación exp(-delta/T) no cambia con la
# escala (delta y T se dividen entre lo mismo), así que el SA se comporta igual que en la Tarea 2.
SA_T0 = 0.10                                 # temperatura inicial
SA_T_FINAL = 0.0001                          # temperatura final
SA_ITER = 30_000                             # iteraciones (movimientos 2-opt propuestos)
SA_ESQUEMA = "exponencial: T_k = T0 * (T_final / T0) ** (k / N)"

# [REQUISITO: «Define ... la regla de empates»]
# Los nombres de las clases son el sufijo de las columnas de costo. El ORDEN de esta lista es:
#   1) el orden de las columnas de salida   2) la prioridad de desempate.
# Regla: si varios algoritmos alcanzan el costo mínimo (costos iguales tras redondear a DECIMALES decimales),
# gana el primero de la lista. Así, 2-opt (la búsqueda local básica, el baseline) conserva los empates, y SA o
# MST+2-opt solo se etiquetan como ganadores si lo superan ESTRICTAMENTE. Entre SA y MST+2-opt el orden es
# convencional (orden de las columnas), lo que además hace que idxmin/argmin den la misma etiqueta.
ALGORITMOS = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
COLUMNAS_COSTO = {a: f"cost_{a}" for a in ALGORITMOS}
COL_ETIQUETA = "best_algorithm"
DECIMALES = 9                                # tolerancia de empate: 1e-9 (absorbe el ruido de punto flotante)

SEMILLA_BASE = 42                            # semilla maestra (reproducibilidad)
COLUMNAS_BASE = ["instance_id", "n", "distribution", "mean_pairwise_dist",
                 "std_coords", "mean_centroid_dist", "mean_nn_dist"]


# =============================================================================
# ADAPTADORES al módulo tsp_algorithms.py
# [REQUISITO: usar los tres algoritmos de la Tarea 2]
# Funciones que se usan de tsp_algorithms.py (no se modifica nada de ese archivo):
#     dist_matrix(points)                                                     -> matriz de distancias
#     tour_length(points, route)                                              -> longitud del ciclo cerrado
#     two_opt(points, route, D=None)                                          -> tour (hill climbing 2-opt, mejora primera)
#     simulated_annealing(points, initial, n_iter, seed, t0, t_final, D=None) -> tour (SA + 2-opt, exponencial)
#     mst_2opt(points, D=None)                                                -> tour (MST con Prim + preorden + 2-opt)
# Esas funciones devuelven solo el tour; cada adaptador regresa (tour, costo) para el resto del script.
# Los algoritmos que arrancan de una solución inicial la reciben del adaptador: un tour ALEATORIO armado con rng.
# =============================================================================
def _correr_2opt(puntos, rng, T0, T_final, n_iter):
    n = len(puntos)
    D = alg.dist_matrix(puntos)
    inicio = rng.sample(range(n), n)                         # solución inicial aleatoria
    tour = alg.two_opt(puntos, inicio, D)
    return tour, alg.tour_length(puntos, tour)


def _correr_sa(puntos, rng, T0, T_final, n_iter):
    n = len(puntos)
    D = alg.dist_matrix(puntos)
    inicio = rng.sample(range(n), n)                         # solución inicial aleatoria
    semilla_sa = rng.getrandbits(32)                         # semilla entera para el RNG interno del SA
    tour = alg.simulated_annealing(puntos, inicio, n_iter, semilla_sa, T0, T_final, D)
    return tour, alg.tour_length(puntos, tour)


def _correr_mst(puntos, rng, T0, T_final, n_iter):
    tour = alg.mst_2opt(puntos, alg.dist_matrix(puntos))     # determinista: no usa rng
    return tour, alg.tour_length(puntos, tour)


EJECUTORES = {"2opt_hill_climbing": _correr_2opt, "sa_2opt": _correr_sa, "mst_2opt": _correr_mst}


# =============================================================================
# LECTURA Y VALIDACIÓN
# =============================================================================
def _es_numero(texto):
    try:
        float(texto)
        return True
    except ValueError:
        return False


def cargar_puntos(ruta):
    """[REQUISITO: cargar las coordenadas desde data/instances/<instance_id>.csv]
    Acepta archivos con encabezado (x, y) o sin encabezado (dos primeras columnas numéricas)."""
    df = pd.read_csv(ruta)
    columnas = {str(c).strip().lower(): c for c in df.columns}
    if "x" in columnas and "y" in columnas:
        xs, ys = df[columnas["x"]], df[columnas["y"]]
    else:
        if all(_es_numero(c) for c in df.columns):           # no había encabezado: la 1a fila era un punto
            df = pd.read_csv(ruta, header=None)
        num = df.select_dtypes("number")
        if num.shape[1] != 2:                  # sin columnas x, y solo se acepta un archivo de 2 columnas numéricas
            raise ValueError(f"{ruta}: se esperaban columnas x, y (o exactamente 2 columnas numéricas); "
                             f"el archivo tiene {list(df.columns)}")
        xs, ys = num.iloc[:, 0], num.iloc[:, 1]
    puntos = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in puntos):
        raise ValueError(f"{ruta}: hay coordenadas no finitas")
    return puntos


def longitud_tour(puntos, tour):
    n = len(tour)
    return sum(math.dist(puntos[tour[i]], puntos[tour[(i + 1) % n]]) for i in range(n))


def validar_resultado(algoritmo, instance_id, puntos, resultado):
    """Revisa lo que devuelve el módulo 'tsp_algorithms' y recalcula el costo con una función propia:
    el costo que se registra es el del tour devuelto, nunca un valor no verificado."""
    tour, costo_reportado = resultado
    if sorted(tour) != list(range(len(puntos))):
        raise ValueError(f"{algoritmo} en {instance_id}: el tour no es una permutación de las {len(puntos)} ciudades")
    costo = longitud_tour(puntos, tour)
    if abs(costo - costo_reportado) > 1e-8 * max(1.0, costo):
        raise ValueError(f"{algoritmo} en {instance_id}: costo reportado {costo_reportado} != costo real {costo}")
    return costo


def elegir_mejor(costos):
    """[REQUISITO: best_algorithm = el de menor costo; empates según la regla documentada]
    Regresa (algoritmo_ganador, cuántos_algoritmos_empatan_en_el_mínimo)."""
    minimo = min(costos[a] for a in ALGORITMOS)
    empatados = [a for a in ALGORITMOS if costos[a] == minimo]      # costos ya redondeados: igualdad exacta
    return empatados[0], len(empatados)


# =============================================================================
# TRABAJO POR INSTANCIA (función de nivel superior para poder usar multiprocessing)
# =============================================================================
def etiquetar_instancia(tarea):
    instance_id, n_esperado, ruta, semilla, T0, T_final, n_iter = tarea
    try:
        puntos = cargar_puntos(ruta)
        if len(puntos) != n_esperado:
            raise ValueError(f"el archivo tiene {len(puntos)} puntos pero train/test dice n = {n_esperado}")
        costos, tiempos = {}, {}
        for a in ALGORITMOS:
            # [REQUISITO: correr los tres algoritmos de Tarea 2 sobre cada instancia]
            # Semilla propia por (semilla base, instancia, algoritmo): resultados reproducibles y no
            # dependen del orden de las instancias ni del número de procesos.
            rng = random.Random(f"{semilla}|{instance_id}|{a}")
            t0 = time.perf_counter()
            resultado = EJECUTORES[a](puntos, rng, T0, T_final, n_iter)
            tiempos[a] = time.perf_counter() - t0
            # [REQUISITO: registrar el costo del tour que obtiene cada algoritmo]
            costos[a] = round(validar_resultado(a, instance_id, puntos, resultado), DECIMALES)
        return instance_id, costos, tiempos
    except Exception as e:                                           # agrega contexto al error
        raise RuntimeError(f"[{instance_id}] {type(e).__name__}: {e}") from e


# =============================================================================
# ENTRADAS, SALIDAS Y RESUMEN
# =============================================================================
def leer_y_verificar(dir_datos):
    train = pd.read_csv(dir_datos / "train.csv")
    test = pd.read_csv(dir_datos / "test.csv")
    for nombre, df in (("train.csv", train), ("test.csv", test)):
        faltan = [c for c in COLUMNAS_BASE if c not in df.columns]
        if faltan:
            raise ValueError(f"{nombre} no tiene las columnas esperadas: {faltan}")
        ya = [c for c in list(COLUMNAS_COSTO.values()) + [COL_ETIQUETA] if c in df.columns]
        if ya:
            raise ValueError(f"{nombre} ya contiene columnas de etiquetas {ya}: se debe partir del archivo SIN etiquetar")
        if df["instance_id"].duplicated().any():
            raise ValueError(f"{nombre} tiene instance_id repetidos")
    comunes = set(train["instance_id"]) & set(test["instance_id"])
    if comunes:
        raise ValueError(f"{len(comunes)} instancias aparecen en train y en test (p. ej. {sorted(comunes)[:3]})")
    return train, test


def etiquetar_dataframe(df, resultados):
    """Agrega a una copia del DataFrame original las 3 columnas de costo y best_algorithm."""
    out = df.copy()
    for a in ALGORITMOS:
        out[COLUMNAS_COSTO[a]] = [resultados[i][0][a] for i in df["instance_id"]]
    out[COL_ETIQUETA] = [elegir_mejor(resultados[i][0])[0] for i in df["instance_id"]]
    assert out[list(COLUMNAS_COSTO.values()) + [COL_ETIQUETA]].notna().all().all()
    assert len(out) == len(df)
    return out


def construir_resumen(train_l, test_l, resultados, parametros, segundos_totales, prueba):
    todo = pd.concat([train_l, test_l], ignore_index=True)
    total = len(todo)
    L = []
    L.append("RESUMEN DEL ETIQUETADO" + ("  *** MODO PRUEBA: NO USAR ESTOS CSV ***" if prueba else ""))
    L.append("=" * 74)
    L.append("Parámetros de SA y del etiquetado")
    for k, v in parametros.items():
        L.append(f"  {k:28s}: {v}")
    L.append("")
    L.append(f"Balance de clases: algoritmo ganador sobre las {total} instancias")
    L.append(f"  {'algoritmo':22s} {'instancias':>10s} {'% de ' + str(total):>9s} {'train':>7s} {'test':>7s}")
    for a in ALGORITMOS:
        c, ctr, cte = (todo[COL_ETIQUETA] == a).sum(), (train_l[COL_ETIQUETA] == a).sum(), (test_l[COL_ETIQUETA] == a).sum()
        L.append(f"  {a:22s} {c:10d} {100 * c / total:8.1f}% {ctr:7d} {cte:7d}")
    mayoritaria = train_l[COL_ETIQUETA].value_counts().reindex(ALGORITMOS).fillna(0).idxmax()
    acc = (test_l[COL_ETIQUETA] == mayoritaria).mean()
    L.append("")
    L.append("Baseline trivial («siempre predecir el algoritmo más frecuente»)")
    L.append(f"  Clase más frecuente en train : {mayoritaria}  ({100 * (train_l[COL_ETIQUETA] == mayoritaria).mean():.1f}% de train)")
    L.append(f"  Accuracy del baseline en test: {100 * acc:.1f}%")
    L.append("")
    niveles = [elegir_mejor(resultados[i][0])[1] for i in todo["instance_id"]]
    L.append("Empates (varios algoritmos con el mismo costo mínimo)")
    L.append(f"  Sin empate          : {sum(x == 1 for x in niveles)} instancias")
    L.append(f"  Empate de 2         : {sum(x == 2 for x in niveles)} instancias")
    L.append(f"  Empate de los 3     : {sum(x == 3 for x in niveles)} instancias")
    L.append(f"  Total con empate    : {sum(x > 1 for x in niveles)} instancias ({100 * sum(x > 1 for x in niveles) / total:.1f}%)")
    L.append("")
    L.append("Ganadores por tamaño n (conteo)")
    tabla = pd.crosstab(todo["n"], todo[COL_ETIQUETA]).reindex(columns=ALGORITMOS, fill_value=0)
    L.append("  " + tabla.to_string().replace("\n", "\n  "))
    L.append("")
    L.append("Tiempo medio por instancia (s)")
    for a in ALGORITMOS:
        L.append(f"  {a:22s}: {sum(resultados[i][1][a] for i in resultados) / len(resultados):.3f}")
    L.append(f"Tiempo total del etiquetado   : {segundos_totales / 60:.1f} min")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Etiqueta train.csv y test.csv corriendo los 3 algoritmos de la Tarea 2.")
    ap.add_argument("--data", default="hh_tsp_dataset/data", help="carpeta con train.csv, test.csv e instances/")
    ap.add_argument("--out", default=None, help="carpeta de salida (por defecto, la misma que --data)")
    ap.add_argument("--seed", type=int, default=SEMILLA_BASE)
    ap.add_argument("--workers", type=int, default=1, help="procesos en paralelo (no cambia los resultados)")
    ap.add_argument("--limit", type=int, default=None, help="SOLO PRUEBA: procesa N filas por archivo y escribe *_PRUEBA")
    args = ap.parse_args()

    dir_datos = Path(args.data)
    dir_salida = Path(args.out) if args.out else dir_datos
    if not (dir_datos / "train.csv").exists():
        sys.exit(f"No encuentro {dir_datos / 'train.csv'}. Usa --data para indicar la carpeta del dataset.")
    dir_salida.mkdir(parents=True, exist_ok=True)
    prueba = args.limit is not None

    train, test = leer_y_verificar(dir_datos)
    if prueba:
        train, test = train.head(args.limit), test.head(args.limit)
    print(f"train: {len(train)} filas | test: {len(test)} filas | total: {len(train) + len(test)} instancias")
    print(f"SA: T0 = {SA_T0}, T_final = {SA_T_FINAL}, iteraciones = {SA_ITER}, {SA_ESQUEMA}")

    todas = pd.concat([train, test], ignore_index=True)
    tareas = [(r.instance_id, int(r.n), str(dir_datos / "instances" / f"{r.instance_id}.csv"),
               args.seed, SA_T0, SA_T_FINAL, SA_ITER) for r in todas.itertuples(index=False)]
    faltantes = [t[0] for t in tareas if not Path(t[2]).exists()]
    if faltantes:
        sys.exit(f"Faltan {len(faltantes)} archivos de instancias (p. ej. {faltantes[:3]}) en {dir_datos / 'instances'}")

    t_ini = time.perf_counter()
    resultados = {}
    if args.workers > 1:
        with Pool(args.workers) as pool:
            iterador = pool.imap(etiquetar_instancia, tareas, chunksize=1)
            for k, (iid, costos, tiempos) in enumerate(iterador, 1):
                resultados[iid] = (costos, tiempos)
                if k % 25 == 0 or k == len(tareas):
                    print(f"  {k}/{len(tareas)} instancias etiquetadas")
    else:
        for k, tarea in enumerate(tareas, 1):
            iid, costos, tiempos = etiquetar_instancia(tarea)
            resultados[iid] = (costos, tiempos)
            if k % 25 == 0 or k == len(tareas):
                print(f"  {k}/{len(tareas)} instancias etiquetadas")
    segundos = time.perf_counter() - t_ini

    train_l, test_l = etiquetar_dataframe(train, resultados), etiquetar_dataframe(test, resultados)
    sufijo = "_PRUEBA" if prueba else ""
    ruta_tr, ruta_te = dir_salida / f"train_labeled{sufijo}.csv", dir_salida / f"test_labeled{sufijo}.csv"
    train_l.to_csv(ruta_tr, index=False)                 # [REQUISITO: escribir los CSV etiquetados]
    test_l.to_csv(ruta_te, index=False)

    parametros = {"SA T0": SA_T0, "SA T_final": SA_T_FINAL, "SA enfriamiento": SA_ESQUEMA,
                  "SA iteraciones": SA_ITER, "2-opt (búsqueda local)": "hill climbing de mejora primera, inicio aleatorio",
                  "MST + 2-opt": "Prim + preorden + 2-opt", "semilla base": args.seed,
                  "regla de empates": " > ".join(ALGORITMOS) + f" (costos redondeados a {DECIMALES} decimales)"}
    resumen = construir_resumen(train_l, test_l, resultados, parametros, segundos, prueba)
    (dir_salida / f"labeling_summary{sufijo}.txt").write_text(resumen + "\n", encoding="utf-8")
    print("\n" + resumen)
    print(f"\nArchivos escritos:\n  {ruta_tr}\n  {ruta_te}\n  {dir_salida / ('labeling_summary' + sufijo + '.txt')}")


if __name__ == "__main__":      # necesario para multiprocessing en Windows/macOS
    main()
