"""Los tres algoritmos de la Tarea 2, reutilizados tal cual (misma lógica y mismo uso del RNG).

Única diferencia: el cambio de costo de un movimiento 2-opt se calcula en O(1) (solo cambian
2 aristas) en vez de recalcular todo el tour en O(n). Los resultados son equivalentes y las
instancias de n=100 corren mucho más rápido.
"""
import math
import random


def tour_length(points, route) -> float:
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def two_opt_move(route, i, k):
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def _delta(D, route, i, k):
    """Cambio de longitud al invertir route[i..k] (i>=1)."""
    n = len(route)
    a, b, c, d = route[i - 1], route[i], route[k], route[(k + 1) % n]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def dist_matrix(points):
    return [[math.dist(p, q) for q in points] for p in points]


def two_opt(points, route, D=None):
    """Hill climbing 2-opt (mejora primera, repite hasta óptimo local)."""
    D = D or dist_matrix(points)
    n = len(route)
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                if _delta(D, best, i, k) < -1e-9:
                    best = two_opt_move(best, i, k)
                    improved = True
    return best


def simulated_annealing(points, initial, n_iter, seed, t0, t_final, D=None):
    """SA con vecindario 2-opt y enfriamiento exponencial (el esquema de la Tarea 2)."""
    D = D or dist_matrix(points)
    n = len(initial)
    rng = random.Random(seed)
    current, best = initial[:], initial[:]
    cur_len = best_len = tour_length(points, current)
    for t in range(1, n_iter + 1):
        T = t0 * (t_final / t0) ** (t / n_iter)
        i = rng.randint(1, n - 2)
        k = rng.randint(i + 1, n - 1)
        delta_E = -_delta(D, current, i, k)          # = cur_len - nxt_len
        if delta_E > 0 or rng.random() < math.exp(delta_E / T):
            current = two_opt_move(current, i, k)
            cur_len -= delta_E
        if cur_len < best_len - 1e-12:
            best, best_len = current[:], cur_len
    return best


def mst_prim(points):
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root=0):
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def mst_2opt(points, D=None):
    return two_opt(points, preorder(mst_prim(points)), D)


if __name__ == "__main__":
    # Prueba rápida: corre los 3 algoritmos sobre una instancia y valida que cada tour sea correcto.
    import csv
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/instances/inst_0001.csv"
    with open(path) as f:
        r = csv.reader(f); next(r)
        pts = [(float(x), float(y)) for x, y in r]
    n = len(pts)
    D = dist_matrix(pts)
    start = list(range(n)); random.Random(1).shuffle(start)
    tours = {
        "2-opt (hill climbing)": two_opt(pts, start, D),
        "SA + 2-opt": simulated_annealing(pts, start, max(3000, 500 * n), 1, 1.0, 1e-4, D),
        "MST + 2-opt": mst_2opt(pts, D),
    }
    print(f"Instancia: {path}  (n = {n})")
    for name, t in tours.items():
        assert sorted(t) == list(range(n)), f"{name}: el tour no visita cada ciudad exactamente una vez"
        print(f"{name:24s} costo = {tour_length(pts, t):.4f}")
    print("OK: los 3 tours son válidos.")
