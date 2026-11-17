"""
train_eval.py  --  PARTE 1 (Persona 3: Modelos)

Entrena un Arbol de Decision y un KNN para predecir `best_algorithm`,
con busqueda de hiperparametros (GridSearchCV) y validacion cruzada
estratificada SOLO sobre train_labeled.csv. El test NO se toca aqui.

Salidas (carpeta results/), que usan las Personas 4 y 5:
    results/model_tree.joblib      pipeline completo del arbol (ya entrenado)
    results/model_knn.joblib       pipeline completo del KNN (ya entrenado)
    results/best_model.joblib      el mejor de los dos segun CV
    results/best_model_name.txt    "tree" o "knn"
    results/cv_results_tree.csv    todas las combinaciones probadas (arbol)
    results/cv_results_knn.csv     todas las combinaciones probadas (KNN)
    results/model_selection.json   mejores hiperparametros, accuracy CV, baseline CV
    results/class_balance_train.csv

Uso:
    python train_eval.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Configuracion compartida (las Personas 4 y 5 pueden importar esto:
#   from train_eval import FEATURES_NUM, FEATURES_CAT, TARGET, load_data)
# ----------------------------------------------------------------------
# Rutas relativas a ESTE archivo (funciona sin importar desde que carpeta se ejecute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train_labeled.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_labeled.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TARGET = "best_algorithm"
FEATURES_NUM = ["n", "mean_pairwise_dist", "std_coords",
                "mean_centroid_dist", "mean_nn_dist"]
FEATURES_CAT = ["distribution"]
FEATURES = FEATURES_NUM + FEATURES_CAT
COST_COLS = ["cost_2opt_hill_climbing", "cost_sa_2opt", "cost_mst_2opt"]

SEED = 42
N_SPLITS = 5


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")
    return df


def build_pipeline(model, scale_numeric):
    """Preprocesamiento + modelo en un solo Pipeline (evita fuga de datos en CV)."""
    num_tf = StandardScaler() if scale_numeric else "passthrough"
    pre = ColumnTransformer([
        ("num", num_tf, FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def train_models():
    """PARTE 1 (Persona 3): entrena, valida con CV y guarda modelos en results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train = load_data(TRAIN_CSV)
    X_train, y_train = train[FEATURES], train[TARGET]

    balance = (y_train.value_counts(normalize=True) * 100).round(1)
    counts = y_train.value_counts()
    bal_df = pd.DataFrame({"count": counts, "percent": balance})
    bal_df.to_csv(os.path.join(RESULTS_DIR, "class_balance_train.csv"))
    print("Balance de clases en train:\n", bal_df, "\n")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    dummy = DummyClassifier(strategy="most_frequent")
    base_scores = cross_val_score(dummy, X_train, y_train, cv=cv, scoring="accuracy")
    baseline_cv = float(base_scores.mean())
    print(f"Baseline CV (clase mayoritaria): {baseline_cv:.4f}\n")

    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    tree_pipe = build_pipeline(DecisionTreeClassifier(random_state=SEED),
                               scale_numeric=False)
    tree_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__class_weight": [None, "balanced"],
    }
    tree_gs = GridSearchCV(tree_pipe, tree_grid, cv=cv, scoring=scoring,
                           refit="accuracy", n_jobs=-1, return_train_score=False)
    tree_gs.fit(X_train, y_train)

    knn_pipe = build_pipeline(KNeighborsClassifier(), scale_numeric=True)
    knn_grid = {
        "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2], 
    }
    knn_gs = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring=scoring,
                          refit="accuracy", n_jobs=-1, return_train_score=False)
    knn_gs.fit(X_train, y_train)

    for name, gs in [("tree", tree_gs), ("knn", knn_gs)]:
        res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_accuracy")
        res.to_csv(os.path.join(RESULTS_DIR, f"cv_results_{name}.csv"), index=False)

    def summarize(gs):
        i = gs.best_index_
        return {
            "best_params": {k.replace("clf__", ""): (None if v is None else
                            (v.item() if hasattr(v, "item") else v))
                            for k, v in gs.best_params_.items()},
            "cv_accuracy_mean": float(gs.cv_results_["mean_test_accuracy"][i]),
            "cv_accuracy_std": float(gs.cv_results_["std_test_accuracy"][i]),
            "cv_f1_macro_mean": float(gs.cv_results_["mean_test_f1_macro"][i]),
        }

    summary = {
        "n_splits": N_SPLITS,
        "seed": SEED,
        "baseline_cv_accuracy": baseline_cv,
        "tree": summarize(tree_gs),
        "knn": summarize(knn_gs),
    }
    best_name = ("tree" if summary["tree"]["cv_accuracy_mean"]
                 >= summary["knn"]["cv_accuracy_mean"] else "knn")
    summary["best_model"] = best_name
    best_est = tree_gs.best_estimator_ if best_name == "tree" else knn_gs.best_estimator_

    joblib.dump(tree_gs.best_estimator_, os.path.join(RESULTS_DIR, "model_tree.joblib"))
    joblib.dump(knn_gs.best_estimator_, os.path.join(RESULTS_DIR, "model_knn.joblib"))
    joblib.dump(best_est, os.path.join(RESULTS_DIR, "best_model.joblib"))
    with open(os.path.join(RESULTS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    with open(os.path.join(RESULTS_DIR, "model_selection.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Arbol ===")
    print(summary["tree"])
    print("=== KNN ===")
    print(summary["knn"])
    print(f"\nMejor modelo segun CV: {best_name}")


# ======================================================================
# PARTE 2 (Persona 4): metricas en test
# Escribir aqui adentro: accuracy vs baseline, matriz de confusion, regret.
# NO reentrenar: solo cargar los modelos guardados en results/.
# ======================================================================
def evaluate_models():
    """PARTE 2 (Persona 4): metricas en test. NO reentrena: carga results/*.joblib.

    Calcula, para el arbol, el KNN y el baseline (clase mayoritaria de TRAIN):
      - accuracy en test
      - matriz de confusion (CSV + PNG)
      - regret absoluto y relativo promedio (costo del algoritmo elegido
        menos costo del mejor algoritmo)
    Salidas en results/: test_metrics.csv, test_metrics.json,
      confusion_<modelo>.csv / .png, regret_per_instance.csv
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, f1_score

    train = load_data(TRAIN_CSV)
    test = load_data(TEST_CSV)
    X_test, y_test = test[FEATURES], test[TARGET]

    classes = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
    cost_of = {c: f"cost_{c}" for c in classes}
    miss = [c for c in cost_of.values() if c not in test.columns]
    if miss:
        raise ValueError(f"Faltan columnas de costo en test: {miss}")

    costs = test[[cost_of[c] for c in classes]].to_numpy(dtype=float)
    best_cost = costs.min(axis=1)

    # Baseline: SIEMPRE predice la clase mas frecuente en TRAIN
    majority = train[TARGET].value_counts().idxmax()

    preds = {
        "baseline": pd.Series([majority] * len(test), index=test.index),
        "tree": pd.Series(joblib.load(os.path.join(RESULTS_DIR, "model_tree.joblib"))
                          .predict(X_test), index=test.index),
        "knn": pd.Series(joblib.load(os.path.join(RESULTS_DIR, "model_knn.joblib"))
                         .predict(X_test), index=test.index),
    }
    with open(os.path.join(RESULTS_DIR, "best_model_name.txt")) as f:
        best_name = f.read().strip()

    rows, regret_table = [], pd.DataFrame({"instance_id": test.get("instance_id", test.index)})
    for name, pred in preds.items():
        chosen_cost = np.array([costs[i, classes.index(p)] for i, p in enumerate(pred)])
        regret = chosen_cost - best_cost
        rel_regret = regret / best_cost * 100.0
        # Acierto "con empates": el algoritmo elegido iguala el costo del mejor
        tie_ok = np.isclose(chosen_cost, best_cost)
        rows.append({
            "model": name,
            "accuracy": float((pred == y_test).mean()),
            "accuracy_tie_aware": float(tie_ok.mean()),
            "f1_macro": float(f1_score(y_test, pred, labels=classes, average="macro",
                                       zero_division=0)),
            "mean_regret": float(regret.mean()),
            "mean_regret_pct": float(rel_regret.mean()),
            "median_regret_pct": float(np.median(rel_regret)),
            "max_regret_pct": float(rel_regret.max()),
            "zero_regret_share": float((regret <= 1e-9).mean()),
        })
        regret_table[f"pred_{name}"] = pred.values
        regret_table[f"regret_{name}"] = regret

        # Matriz de confusion (filas = real, columnas = predicho)
        cm = confusion_matrix(y_test, pred, labels=classes)
        pd.DataFrame(cm, index=[f"real_{c}" for c in classes],
                     columns=[f"pred_{c}" for c in classes]).to_csv(
            os.path.join(RESULTS_DIR, f"confusion_{name}.csv"))
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(classes, rotation=25, ha="right"); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusion - {name} (test)")
        thr = cm.max() / 2 if cm.max() else 0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"confusion_{name}.png"), dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows).set_index("model")
    metrics.to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"))
    regret_table.to_csv(os.path.join(RESULTS_DIR, "regret_per_instance.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({"n_test": int(len(test)), "baseline_class": majority,
                   "best_model_cv": best_name,
                   "metrics": metrics.round(6).to_dict(orient="index")},
                  f, indent=2, ensure_ascii=False)

    print(f"[evaluate_models] n_test = {len(test)} | baseline predice siempre '{majority}'")
    print(f"[evaluate_models] mejor modelo segun CV: {best_name}\n")
    print(metrics.round(4).to_string())
    print("\nLos CSV/PNG quedaron en results/")


# ======================================================================
# PARTE 3 (Persona 5): analisis
# Importancia de caracteristicas (arbol y permutacion), patrones por
# distribucion y tamano, y datos para responder las 3 preguntas del PIA.
# NO reentrena: solo carga los modelos guardados en results/.
#
# Salidas (results/):
#   feature_importance.csv / .png      importancia (arbol + permutacion en test)
#   patterns_by_distribution.csv       ganador por distribucion
#   patterns_by_size.csv               ganador por tamano n
#   patterns_by_dist_size.csv          ganador por (distribucion, n)
#   patterns_strict_by_dist_size.csv   igual, pero solo victorias SIN empate
#   relative_gap_by_size.csv           brecha relativa media de cada algoritmo
#   patterns_heatmaps.png, winners_by_size.png
#   class_recall_test.csv              recall/precision por clase en test
#   analysis_summary.txt               resumen en texto (copiar al reporte)
# ======================================================================
ALGS = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
DISTS = ["uniform", "clustered", "border"]


def analyze_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import classification_report

    os.makedirs(RESULTS_DIR, exist_ok=True)
    train = load_data(TRAIN_CSV)
    test = load_data(TEST_CSV)
    full = pd.concat([train, test], ignore_index=True)
    X_test, y_test = test[FEATURES], test[TARGET]

    tree = joblib.load(os.path.join(RESULTS_DIR, "model_tree.joblib"))
    knn = joblib.load(os.path.join(RESULTS_DIR, "model_knn.joblib"))
    with open(os.path.join(RESULTS_DIR, "best_model_name.txt")) as f:
        best_name = f.read().strip()
    models = {"tree": tree, "knn": knn}

    lines = []

    def say(txt=""):
        print(txt)
        lines.append(txt)

    def path(name):
        return os.path.join(RESULTS_DIR, name)

    say("ANALISIS (Persona 5)")
    say("=" * 70)
    say(f"Mejor modelo segun CV: {best_name}")

    # ------------------------------------------------------------------
    # A) Importancia de caracteristicas
    # ------------------------------------------------------------------
    # A1. Importancia por impureza del arbol (la OHE de distribution se suma)
    names = tree.named_steps["pre"].get_feature_names_out()
    raw = tree.named_steps["clf"].feature_importances_
    imp = {}
    for nm, v in zip(names, raw):
        key = "distribution" if nm.startswith("cat__distribution") else nm.replace("num__", "")
        imp[key] = imp.get(key, 0.0) + float(v)
    imp_tree = pd.Series(imp).reindex(FEATURES).fillna(0.0)

    # A2. Importancia por permutacion en test (accuracy), sobre las columnas
    #     originales (la permutacion de 'distribution' mueve toda la categoria).
    perm = {}
    for nm, mdl in models.items():
        r = permutation_importance(mdl, X_test, y_test, scoring="accuracy",
                                   n_repeats=100, random_state=SEED, n_jobs=1)
        perm[nm] = (pd.Series(r.importances_mean, index=FEATURES),
                    pd.Series(r.importances_std, index=FEATURES))

    fi = pd.DataFrame({
        "tree_impureza": imp_tree,
        "perm_tree_mean": perm["tree"][0], "perm_tree_std": perm["tree"][1],
        "perm_knn_mean": perm["knn"][0], "perm_knn_std": perm["knn"][1],
    })
    fi.index.name = "feature"
    fi.sort_values("perm_knn_mean", ascending=False).to_csv(path("feature_importance.csv"))

    say("\nA) IMPORTANCIA DE CARACTERISTICAS")
    say(f"Arbol: profundidad={tree.named_steps['clf'].get_depth()}, "
        f"hojas={tree.named_steps['clf'].get_n_leaves()}")
    say("Importancia por impureza (arbol, suma = 1):")
    for k, v in imp_tree.sort_values(ascending=False).items():
        say(f"   {k:20s} {v:.3f}")
    say("Importancia por permutacion en test (caida de accuracy, media +- std de 100 repeticiones):")
    for nm in ["tree", "knn"]:
        say(f"  Modelo {nm}:")
        m, sd = perm[nm]
        for k in m.sort_values(ascending=False).index:
            say(f"   {k:20s} {m[k]:+.4f} +- {sd[k]:.4f}")


    # A3. Reglas del arbol
    from sklearn.tree import export_text
    short = [n.replace("num__", "").replace("cat__", "") for n in names]
    say("\nReglas del arbol (n sin escalar):")
    say(export_text(tree.named_steps["clf"], feature_names=short))

    # A4. Permutacion por GRUPOS: n contra toda la geometria junta
    rng = np.random.default_rng(SEED)
    geo = [c for c in FEATURES if c != "n"]
    groups = {"n": ["n"], "distribution + 4 numericas": geo,
              "4 numericas": FEATURES_NUM[1:]}
    say("Permutacion por grupos en test (caida de accuracy, media +- std de 100 repeticiones):")
    grows = []
    for nm, mdl in models.items():
        base = (mdl.predict(X_test) == y_test).mean()
        for gname, cols in groups.items():
            drops = []
            for _ in range(100):
                idx = rng.permutation(len(X_test))
                Xp = X_test.copy()
                for c in cols:
                    Xp[c] = X_test[c].to_numpy()[idx]
                drops.append(base - (mdl.predict(Xp) == y_test).mean())
            say(f"   {nm:5s} {gname:28s} {np.mean(drops):+.4f} +- {np.std(drops):.4f}")
            grows.append({"model": nm, "group": gname,
                          "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops))})
    pd.DataFrame(grows).to_csv(path("permutation_groups.csv"), index=False)

    # A5. Correlacion entre caracteristicas numericas (en las 500 instancias)
    corr = pd.concat([train, test])[FEATURES_NUM].corr(method="spearman").round(2)
    corr.to_csv(path("feature_correlation.csv"))
    say("\nCorrelacion de Spearman entre caracteristicas numericas (500 instancias):")
    say(corr.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    imp_tree.sort_values().plot.barh(ax=ax[0], color="#4C72B0")
    ax[0].set_title("Arbol: importancia por impureza")
    ax[0].set_xlabel("importancia (suma = 1)")
    order = perm["knn"][0].sort_values().index
    y = np.arange(len(order))
    ax[1].barh(y - 0.2, perm["knn"][0][order], 0.4, xerr=perm["knn"][1][order],
               label="KNN", color="#DD8452")
    ax[1].barh(y + 0.2, perm["tree"][0][order], 0.4, xerr=perm["tree"][1][order],
               label="Arbol", color="#4C72B0")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(order)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_title("Importancia por permutacion (test)")
    ax[1].set_xlabel("caida de accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path("feature_importance.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # B) Patrones por distribucion y tamano (las 500 instancias etiquetadas)
    # ------------------------------------------------------------------
    full["dist"] = pd.Categorical(full["distribution"], DISTS, ordered=True)
    costs = full[COST_COLS].round(9)
    full["n_winners"] = (costs.eq(costs.min(axis=1), axis=0)).sum(axis=1)
    full["strict"] = full["n_winners"] == 1
    gap = costs.sub(costs.min(axis=1), axis=0).div(costs.min(axis=1), axis=0) * 100
    gap.columns = ALGS

    def winners_table(df, by):
        idx = [df[b] for b in by] if isinstance(by, list) else df[by]
        ct = pd.crosstab(idx, df[TARGET]).reindex(columns=ALGS, fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        out = ct.join(pct.add_suffix("_%").round(1))
        out["total"] = ct.sum(axis=1)
        return out

    by_dist = winners_table(full, "dist")
    by_n = winners_table(full, "n")
    by_dn = winners_table(full, ["dist", "n"])
    strict_dn = winners_table(full[full["strict"]], ["dist", "n"])
    by_dist.to_csv(path("patterns_by_distribution.csv"))
    by_n.to_csv(path("patterns_by_size.csv"))
    by_dn.to_csv(path("patterns_by_dist_size.csv"))
    strict_dn.to_csv(path("patterns_strict_by_dist_size.csv"))

    gap_n = gap.groupby(full["n"]).mean().round(3)
    gap_n["pct_empate_3"] = (full.groupby("n")["n_winners"].apply(lambda s: (s == 3).mean()) * 100).round(1)
    gap_n.to_csv(path("relative_gap_by_size.csv"))

    say("\nB) PATRONES POR DISTRIBUCION Y TAMANO (500 instancias)")
    say(f"Instancias con empate en el minimo: {(~full['strict']).sum()} "
        f"({(~full['strict']).mean() * 100:.1f}%); empate de los 3: "
        f"{(full['n_winners'] == 3).sum()}")
    say("\nGanador por distribucion (conteo y %):")
    say(by_dist.to_string())
    say("\nGanador por tamano n (conteo y %):")
    say(by_n.to_string())
    say("\nBrecha relativa media (%) de cada algoritmo vs el mejor, por n, "
        "y % de instancias con empate de los 3:")
    say(gap_n.to_string())
    say("\nSolo victorias SIN empate, por distribucion (conteo):")
    say(winners_table(full[full["strict"]], "dist").to_string())
    say("\nSolo victorias SIN empate, por tamano (conteo):")
    say(winners_table(full[full["strict"]], "n").to_string())

    # Figura 1: heatmaps (% de instancias que gana cada algoritmo)
    ns = sorted(full["n"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, alg in zip(axes, ALGS):
        mat = np.full((len(DISTS), len(ns)), np.nan)
        for i, d in enumerate(DISTS):
            for j, n in enumerate(ns):
                sub = full[(full["distribution"] == d) & (full["n"] == n)]
                if len(sub):
                    mat[i, j] = (sub[TARGET] == alg).mean() * 100
        im = ax.imshow(mat, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns, fontsize=9)
        ax.set_yticks(range(len(DISTS)))
        ax.set_yticklabels(DISTS, fontsize=9)
        ax.set_xlabel("n")
        ax.set_title(f"% que gana {alg}", fontsize=9)
        for i in range(len(DISTS)):
            for j in range(len(ns)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                            color="w" if mat[i, j] < 60 else "k", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="%")
    fig.savefig(path("patterns_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figura 2: barras apiladas ganador por n, por distribucion
    colors = {"2opt_hill_climbing": "#4C72B0", "sa_2opt": "#DD8452", "mst_2opt": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, d in zip(axes, DISTS):
        sub = full[full["distribution"] == d]
        ct = pd.crosstab(sub["n"], sub[TARGET]).reindex(columns=ALGS, fill_value=0)
        (ct.div(ct.sum(axis=1), axis=0) * 100).plot.bar(
            stacked=True, ax=ax, color=[colors[a] for a in ALGS], width=0.8, legend=False)
        ax.set_title(d)
        ax.set_xlabel("n")
        ax.set_ylabel("% de instancias")
    axes[0].legend(ALGS, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path("winners_by_size.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # C) Datos para: "el modelo identifica SA / MST o colapsa a 2-opt?"
    # ------------------------------------------------------------------
    say("\nC) DETECCION DE CLASES MINORITARIAS EN TEST")
    rows = []
    for nm, mdl in models.items():
        pred = pd.Series(mdl.predict(X_test), index=test.index)
        rep = classification_report(y_test, pred, labels=ALGS, output_dict=True,
                                    zero_division=0)
        say(f"\nModelo {nm}{' (mejor segun CV)' if nm == best_name else ''}: "
            f"accuracy = {(pred == y_test).mean():.4f}")
        say("  Predicciones por clase: " +
            ", ".join(f"{a}={int((pred == a).sum())}" for a in ALGS))
        say("  Reales por clase:       " +
            ", ".join(f"{a}={int((y_test == a).sum())}" for a in ALGS))
        for a in ALGS:
            r = rep[a]
            say(f"  {a:20s} precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
                f"f1={r['f1-score']:.3f}  (soporte={int(r['support'])})")
            rows.append({"model": nm, "class": a, "precision": r["precision"],
                         "recall": r["recall"], "f1": r["f1-score"],
                         "support": int(r["support"])})
        # Accuracy "consciente de empates": el algoritmo elegido alcanza el
        # costo minimo (aunque otro tambien lo alcance).
        tc = test[COST_COLS].round(9)
        col = {"2opt_hill_climbing": COST_COLS[0], "sa_2opt": COST_COLS[1],
               "mst_2opt": COST_COLS[2]}
        chosen = np.array([tc.loc[i, col[p]] for i, p in pred.items()])
        hit = chosen == tc.min(axis=1).to_numpy()
        say(f"  Elige un algoritmo con costo minimo (acepta empates): {hit.mean():.4f}")
        strict_t = (tc.eq(tc.min(axis=1), axis=0)).sum(axis=1) == 1
        say(f"  Accuracy solo en instancias SIN empate ({int(strict_t.sum())}): "
            f"{(pred[strict_t] == y_test[strict_t]).mean():.4f}")

        real_non = y_test != "2opt_hill_climbing"
        pred_non = pred != "2opt_hill_climbing"
        tp = int((real_non & pred_non).sum())
        say(f"  Detecta 'SA o MST superan a 2-opt': recall={tp / real_non.sum():.3f} "
            f"({tp}/{int(real_non.sum())}), precision={tp / max(int(pred_non.sum()), 1):.3f} "
            f"({tp}/{int(pred_non.sum())})")
        rows.append({"model": nm, "class": "no_2opt (SA o MST gana)",
                     "precision": tp / max(int(pred_non.sum()), 1),
                     "recall": tp / real_non.sum(), "f1": np.nan,
                     "support": int(real_non.sum())})
    pd.DataFrame(rows).to_csv(path("class_recall_test.csv"), index=False)

    with open(path("analysis_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[analyze_results] resultados guardados en {RESULTS_DIR}/")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-train", action="store_true",
                    help="solo entrenar (parte 1)")
    ap.add_argument("--only-eval", action="store_true",
                    help="solo evaluar en test con los modelos ya guardados (parte 2)")
    ap.add_argument("--only-analysis", action="store_true",
                    help="solo analisis (parte 3)")
    a = ap.parse_args()

    if a.only_train:
        train_models()
    elif a.only_eval:
        evaluate_models()
    elif a.only_analysis:
        analyze_results()
    else:  # sin banderas: todo en orden
        train_models()
        evaluate_models()
        analyze_results()


if __name__ == "__main__":
    main()