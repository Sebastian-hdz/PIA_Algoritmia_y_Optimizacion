"""
Proyecto Final - Hiper-heurística de Selección para TSP
Verificaciones adicionales que cita el reporte y que no salen de train_eval.py.

QUÉ HACE
    1. Comparación contra políticas fijas y oráculo (sección 4.3 del reporte):
       «siempre 2-opt», «siempre SA+2-opt», «siempre MST+2-opt», árbol, KNN y oráculo.
    2. Intervalo de confianza del 95 % por bootstrap para la diferencia de regret relativo entre
       cada modelo y «siempre SA+2-opt» (sección 4.3).
    3. Verificación independiente de los costos con el óptimo exacto de Held-Karp en las
       instancias con n <= 15 (sección 2.5).
    4. Estadísticas de los errores de cada modelo (sección 4.2).
    5. Figuras 1 y 2 del reporte (accuracy y regret por política).

USO
    python extra_checks.py                  # bootstrap con 10,000 remuestreos y semilla 42
    python extra_checks.py --boot 20000 --seed 7

ENTRADAS   train_labeled.csv, test_labeled.csv, results/regret_per_instance.csv (de train_eval.py --only-eval),
           hh_tsp_dataset/data/instances/<instance_id>.csv
SALIDAS    results/policy_comparison.csv, results/bootstrap_vs_always_sa.csv, results/error_stats.csv,
           results/heldkarp_check.csv, results/heldkarp_summary.txt,
           results/fig_accuracy_test.png, results/fig_regret_policies.png

Requiere: numpy, pandas, matplotlib. Primero hay que haber corrido train_eval.py --only-eval.
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAIN_CSV = os.path.join(BASE_DIR, "train_labeled.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_labeled.csv")
INST_DIR = os.path.join(BASE_DIR, "hh_tsp_dataset", "data", "instances")

ALGS = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
COST = {a: f"cost_{a}" for a in ALGS}
NOMBRES = {"always_2opt": "Siempre 2-opt (baseline)", "always_sa": "Siempre SA+2-opt",
           "always_mst": "Siempre MST+2-opt", "tree": "Árbol de decisión", "knn": "KNN",
           "oracle": "Oráculo (siempre el mejor)"}
TOL = 1e-9


def p(name):
    return os.path.join(RESULTS_DIR, name)


# ---------------------------------------------------------------------------
# 1, 2, 4 y 5: políticas, bootstrap, errores y figuras (conjunto de test)
# ---------------------------------------------------------------------------
def politicas(n_boot, seed):
    test = pd.read_csv(TEST_CSV)
    reg = pd.read_csv(p("regret_per_instance.csv"))
    test = test.merge(reg[["instance_id", "pred_tree", "pred_knn"]], on="instance_id", how="left")
    if test["pred_tree"].isna().any():
        raise SystemExit("regret_per_instance.csv no coincide con test_labeled.csv: "
                         "corre primero python train_eval.py --only-eval")

    costs = test[[COST[a] for a in ALGS]].to_numpy(dtype=float)
    best = costs.min(axis=1)
    label = test["best_algorithm"].to_numpy()

    choices = {
        "always_2opt": np.array(["2opt_hill_climbing"] * len(test)),
        "always_sa": np.array(["sa_2opt"] * len(test)),
        "always_mst": np.array(["mst_2opt"] * len(test)),
        "tree": test["pred_tree"].to_numpy(),
        "knn": test["pred_knn"].to_numpy(),
        "oracle": label,
    }

    rows, rel = [], {}
    for k, pred in choices.items():
        chosen = np.array([costs[i, ALGS.index(a)] for i, a in enumerate(pred)])
        regret = chosen - best
        rel[k] = regret / best * 100.0
        rows.append({
            "policy": k, "nombre": NOMBRES[k],
            "accuracy": float((pred == label).mean()),
            "accuracy_con_empates": float((regret <= TOL).mean()),
            "regret_promedio": float(regret.mean()),
            "regret_relativo_pct": float(rel[k].mean()),
            "regret_cero": float((regret <= TOL).mean()),
            "costo_total_test": float(chosen.sum()),
        })
    tabla = pd.DataFrame(rows)
    tabla.to_csv(p("policy_comparison.csv"), index=False)
    print("\n[1] COMPARACION CONTRA POLITICAS FIJAS Y ORACULO (test, n = %d)" % len(test))
    print(tabla.drop(columns="nombre").round(4).to_string(index=False))

    # --- bootstrap: diferencia de regret relativo (modelo - siempre SA), por instancia ---
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(test), size=(n_boot, len(test)))
    brows = []
    for m in ["tree", "knn"]:
        diff = rel[m] - rel["always_sa"]
        medias = diff[idx].mean(axis=1)
        lo, hi = np.percentile(medias, [2.5, 97.5])
        brows.append({"modelo": m, "diferencia_media_pp": float(diff.mean()),
                      "ic95_inf_pp": float(lo), "ic95_sup_pp": float(hi),
                      "incluye_cero": bool(lo <= 0 <= hi), "remuestreos": n_boot, "semilla": seed})
    boot = pd.DataFrame(brows)
    boot.to_csv(p("bootstrap_vs_always_sa.csv"), index=False)
    print(f"\n[2] BOOTSTRAP (diferencia de regret relativo vs 'siempre SA', {n_boot} remuestreos, semilla {seed})")
    print(boot.round(3).to_string(index=False))

    # --- errores de cada modelo ---
    erows = []
    for m in ["tree", "knn"]:
        err = choices[m] != label
        erows.append({"modelo": m, "errores": int(err.sum()),
                      "errores_con_regret_cero": int((rel[m][err] <= TOL).sum()),
                      "regret_relativo_medio_en_errores_pct": float(rel[m][err].mean())})
    errores = pd.DataFrame(erows)
    errores.to_csv(p("error_stats.csv"), index=False)
    print("\n[4] ERRORES DE CADA MODELO")
    print(errores.round(3).to_string(index=False))

    figuras(tabla)


def figuras(tabla):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tabla.set_index("policy")
    # Figura 1: accuracy exacta vs con empates (baseline, árbol, KNN)
    ks = ["always_2opt", "tree", "knn"]
    etiquetas = ["Baseline\n(siempre 2-opt)", "Árbol", "KNN"]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b1 = ax.bar(x - 0.2, t.loc[ks, "accuracy"] * 100, 0.4, label="Accuracy (etiqueta exacta)", color="#2E6DB4")
    b2 = ax.bar(x + 0.2, t.loc[ks, "accuracy_con_empates"] * 100, 0.4, label="Accuracy con empates",
                color="#2E9E73")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("%")
    ax.set_ylim(0, 90)
    ax.set_title("Accuracy en test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(p("fig_accuracy_test.png"), dpi=150)
    plt.close(fig)

    # Figura 2: regret relativo promedio por política
    ks = ["always_2opt", "always_sa", "always_mst", "tree", "knn", "oracle"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vals = t.loc[ks, "regret_relativo_pct"]
    bars = ax.bar(range(len(ks)), vals, color=["#8A8A8A", "#E08A2E", "#8A8A8A", "#2E6DB4", "#2E9E73", "#444444"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{b.get_height():.2f}%",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["Siempre\n2-opt", "Siempre\nSA", "Siempre\nMST", "Árbol", "KNN", "Oráculo"], fontsize=8)
    ax.set_ylabel("regret relativo promedio (%)")
    ax.set_title("Regret relativo promedio por política (test)")
    fig.tight_layout()
    fig.savefig(p("fig_regret_policies.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3: Held-Karp (óptimo exacto) para n <= 15
# ---------------------------------------------------------------------------
def held_karp(D):
    """Longitud del tour óptimo (ciclo cerrado) por programación dinámica, vectorizada por capas."""
    n = len(D)
    m = n - 1                                   # nodos 1..n-1; el nodo 0 es el origen fijo
    D0 = D[0, 1:]
    Dm = D[1:, 1:]
    full = (1 << m) - 1
    dp = np.full((1 << m, m), np.inf)
    for j in range(m):
        dp[1 << j, j] = D0[j]
    masks = np.arange(1 << m)
    pop = np.zeros(1 << m, dtype=int)
    for j in range(m):
        pop += (masks >> j) & 1
    for s in range(2, m + 1):
        capa = masks[pop == s]
        for j in range(m):
            sel = capa[(capa >> j) & 1 == 1]
            prev = sel ^ (1 << j)
            dp[sel, j] = (dp[prev] + Dm[:, j][None, :]).min(axis=1)
    return float((dp[full] + D0).min())


def cargar_puntos(instance_id):
    df = pd.read_csv(os.path.join(INST_DIR, f"{instance_id}.csv"))
    return df.iloc[:, :2].to_numpy(dtype=float)


def verificar_heldkarp(nmax=15):
    df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
    df = df[df["n"] <= nmax].reset_index(drop=True)
    opt = []
    for iid in df["instance_id"]:
        pts = cargar_puntos(iid)
        D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        opt.append(held_karp(D))
    df["optimo_heldkarp"] = np.round(opt, 9)
    for a in ALGS:
        df[f"brecha_{a}_pct"] = (df[COST[a]] - df["optimo_heldkarp"]) / df["optimo_heldkarp"] * 100.0
    df[["instance_id", "n", "distribution", "optimo_heldkarp"] + [COST[a] for a in ALGS] +
       [f"brecha_{a}_pct" for a in ALGS]].to_csv(p("heldkarp_check.csv"), index=False)

    lineas = [f"VERIFICACION CON HELD-KARP (n <= {nmax}): {len(df)} instancias"]
    for a in ALGS:
        menor = int((df[COST[a]] < df["optimo_heldkarp"] - 1e-8).sum())
        lineas.append(f"  {a:20s} costos por debajo del optimo: {menor} | brecha media: "
                      f"{df[f'brecha_{a}_pct'].mean():.4f} %")
    lineas.append("  (un costo menor que el optimo seria un error; debe dar 0 en los tres algoritmos)")
    texto = "\n".join(lineas)
    print("\n[3] " + texto)
    with open(p("heldkarp_summary.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10_000, help="remuestreos del bootstrap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-heldkarp", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    politicas(a.boot, a.seed)
    if not a.skip_heldkarp:
        verificar_heldkarp()
    print(f"\n[extra_checks] salidas en {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
