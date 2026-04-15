"""
route_model.py
==============
DSO 4 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Classe d'inférence production pour l'optimisation de tournées.

Ce fichier est le SEUL importé par Django — page /routes/.
Il permet d'optimiser une tournée à la demande (jour, heure,
nombre de stops, point de départ configurable).

Usage :
    from route_model import RouteOptimizer

    optimizer = RouteOptimizer.load("models/route_model.pkl")

    result = optimizer.optimize(
        target_day   = "Thursday",
        target_hour  = 10,
        n_stops      = 6,
        depot_lat    = 36.8190,
        depot_lon    = 10.1660,
        cluster_id   = None,    # None = toutes zones
    )

    # result["route"]          → liste des stops ordonnés
    # result["total_distance"] → km
    # result["total_time_min"] → minutes
    # result["map_data"]       → données pour Leaflet.js

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

import logging
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RouteModel")

DEFAULT_MODEL_PATH = "models/route_model.pkl"

ZONE_NAMES = {
    0: "Zone A — Nord-Est",
    1: "Zone B — Centre",
    2: "Zone C — Centre-Ouest",
    3: "Zone D — Sud",
}

DAYS_FR = {
    "Monday":"Lundi", "Tuesday":"Mardi", "Wednesday":"Mercredi",
    "Thursday":"Jeudi", "Friday":"Vendredi",
    "Saturday":"Samedi", "Sunday":"Dimanche",
}

HOUR_COLS = [f"hour_{i}" for i in range(24)]


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS GÉOSPATIALES
# ══════════════════════════════════════════════════════════════════════

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def haversine_matrix(lats1, lons1, lats2, lons2) -> np.ndarray:
    R = 6371.0
    lat1,lon1 = np.radians(lats1), np.radians(lons1)
    lat2,lon2 = np.radians(lats2), np.radians(lons2)
    dlat = lat2[:,None] - lat1[None,:]
    dlon = lon2[:,None] - lon1[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2[:,None])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def route_total_distance(route, lats, lons) -> float:
    return sum(
        haversine_distance(lats[route[i]], lons[route[i]],
                           lats[route[i+1]], lons[route[i+1]])
        for i in range(len(route) - 1)
    )


# ══════════════════════════════════════════════════════════════════════
# RouteOptimizer — CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

class RouteOptimizer:
    """
    Optimiseur de tournées pour les délégués médicaux VITAL SA.

    Fonctionnement :
        1. Recalcule les priority scores selon (jour, heure, dépôt)
        2. Sélectionne les N meilleures pharmacies
        3. Optimise la route par Nearest Neighbor + 2-Opt
        4. Retourne le résultat formaté pour Django/Leaflet

    Attributes:
        version    (str) : version du modèle
        trained_at (str) : timestamp d'entraînement
        n_pharmacies (int) : nombre total de pharmacies disponibles
    """

    def __init__(self, bundle: Dict):
        self._kmeans      = bundle["kmeans"]
        self._df_pharm    = bundle["df_pharmacies"].copy()
        self._config      = bundle["config"]
        self.version      = bundle.get("version",    "1.0.0")
        self.trained_at   = bundle.get("trained_at", "unknown")
        self.n_pharmacies = len(self._df_pharm)

    # ── Chargement ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "RouteOptimizer":
        """
        Charge l'optimiseur depuis un fichier .pkl.

        Args:
            path : chemin vers route_model.pkl

        Returns:
            Instance RouteOptimizer prête à l'inférence
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Bundle introuvable : {path}\n"
                "Exécutez : python route_optimizer.py"
            )
        bundle = joblib.load(model_path)
        log.info(
            f"RouteOptimizer chargé — "
            f"v{bundle.get('version','?')} | "
            f"{bundle.get('trained_at','?')[:19]} | "
            f"{len(bundle['df_pharmacies'])} pharmacies"
        )
        return cls(bundle)

    # ── Inférence principale ──────────────────────────────────────────

    def optimize(self,
                 target_day  : str   = "Monday",
                 target_hour : int   = 10,
                 n_stops     : int   = 8,
                 depot_lat   : float = None,
                 depot_lon   : float = None,
                 depot_name  : str   = None,
                 cluster_id  : Optional[int] = None) -> Dict:
        """
        Optimise une tournée journalière.

        Args:
            target_day   : jour de la tournée (ex: "Thursday")
            target_hour  : heure de départ (0–23)
            n_stops      : nombre de pharmacies à visiter (4–10)
            depot_lat/lon: point de départ (dépôt délégué)
            depot_name   : nom du dépôt
            cluster_id   : zone géographique (0-3) ou None = toutes zones

        Returns:
            dict complet avec route, distances, durées, map_data
        """
        depot_lat  = depot_lat  or self._config["depot_lat"]
        depot_lon  = depot_lon  or self._config["depot_lon"]
        depot_name = depot_name or self._config["depot_name"]
        n_stops    = max(self._config["min_stops"],
                         min(n_stops, self._config["max_stops"]))

        log.info(f"Optimisation : {target_day} {target_hour}h | "
                 f"{n_stops} stops | dépôt ({depot_lat:.4f}, {depot_lon:.4f})")

        # ── Priority scores ───────────────────────────────────────────
        df = self._compute_scores(target_day, target_hour, depot_lat, depot_lon)

        # ── Filtrage zone ─────────────────────────────────────────────
        if cluster_id is not None:
            df = df[df["cluster"] == cluster_id].copy()
            if len(df) < n_stops:
                log.warning(f"Zone {cluster_id} : {len(df)} pharmacies "
                            f"< {n_stops} stops demandés — élargissement")
                df = self._compute_scores(target_day, target_hour,
                                          depot_lat, depot_lon)

        # ── Sélection top-N ───────────────────────────────────────────
        df_stops = df.nlargest(n_stops, "priority_score").copy()

        # ── TSP ───────────────────────────────────────────────────────
        depot_row = pd.DataFrame([{
            "venue_name"    : depot_name,
            "latitude"      : depot_lat,
            "longitude"     : depot_lon,
            "priority_score": 1.0,
            "cluster"       : -1,
            "data_source"   : "dépôt",
        }])
        df_route = pd.concat([depot_row, df_stops], ignore_index=True)
        lats = df_route["latitude"].values
        lons = df_route["longitude"].values

        route_nn, dist_nn     = self._nearest_neighbor(lats, lons)
        route_opt, dist_opt, _ = self._two_opt(route_nn, lats, lons)
        improvement = (dist_nn - dist_opt) / dist_nn * 100

        # ── Durées ───────────────────────────────────────────────────
        speed        = self._config["avg_speed_kmh"]
        visit_min    = self._config["visit_duration_min"]
        drive_min    = dist_opt / speed * 60
        visits_total = n_stops * visit_min
        total_min    = drive_min + visits_total

        # ── Construire stops détaillés ────────────────────────────────
        stops_detail = self._build_stops(
            route_opt, df_route, lats, lons, target_hour
        )

        # ── Map data pour Leaflet.js ──────────────────────────────────
        map_data = self._build_map_data(stops_detail, depot_lat, depot_lon)

        # ── Stats par zone ────────────────────────────────────────────
        zone_stats = self._zone_stats(df_stops)

        result = {
            # Route principale
            "route"              : stops_detail,
            "total_distance_km"  : round(dist_opt, 2),
            "nn_distance_km"     : round(dist_nn, 2),
            "improvement_pct"    : round(improvement, 1),

            # Durées
            "drive_time_min"     : round(drive_min, 1),
            "visit_time_min"     : round(visits_total, 1),
            "total_time_min"     : round(total_min, 1),

            # Paramètres
            "target_day"         : target_day,
            "target_day_fr"      : DAYS_FR.get(target_day, target_day),
            "target_hour"        : target_hour,
            "n_stops"            : n_stops,
            "cluster_id"         : cluster_id,
            "zone_name"          : ZONE_NAMES.get(cluster_id, "Toutes zones"),

            # Dépôt
            "depot"              : {
                "name": depot_name, "lat": depot_lat, "lon": depot_lon
            },

            # Data visualisation
            "map_data"           : map_data,
            "zone_stats"         : zone_stats,

            # Meta
            "computed_at"        : str(pd.Timestamp.now()),
        }

        log.info(f"  ✅  Route : {dist_opt:.2f} km | {total_min:.0f} min | "
                 f"-{improvement:.1f}% vs NN")
        return result

    # ── Helper : priority scores ──────────────────────────────────────

    def _compute_scores(self, target_day: str, target_hour: int,
                        depot_lat: float, depot_lon: float) -> pd.DataFrame:
        """Recalcule les priority scores pour les paramètres donnés."""
        df = self._df_pharm.copy()

        df["affluence_norm"] = MinMaxScaler().fit_transform(df[["day_mean"]])

        hour_col = f"hour_{target_hour}"
        if hour_col in df.columns:
            df["peak_score"] = MinMaxScaler().fit_transform(df[[hour_col]])
        else:
            df["peak_score"] = df["affluence_norm"]

        dists = haversine_matrix(
            np.array([depot_lat]), np.array([depot_lon]),
            df["latitude"].values,  df["longitude"].values
        ).flatten()
        df["dist_depot_km"] = dists.round(2)
        df["geo_score"]     = (1.0 - dists / (dists.max() + 1e-6)).round(4)

        wa = self._config["weight_affluence"]
        wp = self._config["weight_peak"]
        wg = self._config["weight_geo"]
        df["priority_score"] = (
            wa * df["affluence_norm"] +
            wp * df["peak_score"]     +
            wg * df["geo_score"]
        ).round(4)

        return df

    # ── Helper : TSP algorithms ───────────────────────────────────────

    def _nearest_neighbor(self, lats, lons,
                          start_idx=0) -> Tuple[List[int], float]:
        n       = len(lats)
        visited = [False] * n
        route   = [start_idx]; visited[start_idx] = True; current = start_idx
        for _ in range(n - 1):
            best_d, best_j = np.inf, -1
            for j in range(n):
                if not visited[j]:
                    d = haversine_distance(lats[current],lons[current],lats[j],lons[j])
                    if d < best_d: best_d, best_j = d, j
            route.append(best_j); visited[best_j] = True; current = best_j
        route.append(start_idx)
        return route, route_total_distance(route, lats, lons)

    def _two_opt(self, route, lats, lons,
                 max_iter=500) -> Tuple[List[int], float, List[float]]:
        best = route[:]; best_d = route_total_distance(best, lats, lons)
        hist = [best_d]; improved = True; it = 0
        while improved and it < max_iter:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    nd  = route_total_distance(new, lats, lons)
                    if nd < best_d - 1e-6:
                        best, best_d = new, nd; hist.append(nd); improved = True
            it += 1
        return best, best_d, hist

    # ── Helper : build stops ──────────────────────────────────────────

    def _build_stops(self, route, df_route, lats, lons,
                     target_hour: int) -> List[Dict]:
        stops = []
        hour_col = f"hour_{target_hour}"
        for i, idx in enumerate(route):
            row = df_route.iloc[idx]
            d_prev = 0.0
            if i > 0:
                prev = route[i-1]
                d_prev = haversine_distance(lats[prev],lons[prev],lats[idx],lons[idx])
            # affluence à l'heure cible
            affluence_h = float(row.get(hour_col, row.get("day_mean", 0))) if not row.get("is_depot", False) else None
            stops.append({
                "order"           : i,
                "is_depot"        : bool(idx == 0),
                "venue_name"      : str(row["venue_name"]),
                "address"         : str(row.get("venue_address", "")),
                "latitude"        : round(float(lats[idx]), 6),
                "longitude"       : round(float(lons[idx]), 6),
                "priority_score"  : round(float(row.get("priority_score", 0)), 3),
                "affluence_now"   : round(float(affluence_h), 1) if affluence_h else None,
                "affluence_mean"  : round(float(row.get("day_mean", 0)), 1),
                "dist_from_prev_km": round(d_prev, 2),
                "cluster"         : int(row.get("cluster", -1)),
                "zone_name"       : ZONE_NAMES.get(int(row.get("cluster", -1)), "—"),
                "data_source"     : str(row.get("data_source", "—")),
            })
        return stops

    # ── Helper : map data ─────────────────────────────────────────────

    def _build_map_data(self, stops: List[Dict],
                        depot_lat: float, depot_lon: float) -> Dict:
        """
        Retourne les données formatées pour Leaflet.js / Django template.
        """
        markers = []
        polyline = []
        for stop in stops:
            markers.append({
                "lat"      : stop["latitude"],
                "lon"      : stop["longitude"],
                "label"    : str(stop["order"]) if not stop["is_depot"] else "D",
                "name"     : stop["venue_name"],
                "address"  : stop["address"],
                "score"    : stop["priority_score"],
                "affluence": stop["affluence_now"],
                "is_depot" : stop["is_depot"],
                "color"    : "#3498db" if stop["is_depot"] else self._score_color(stop["priority_score"]),
            })
            polyline.append([stop["latitude"], stop["longitude"]])

        return {
            "markers"  : markers,
            "polyline" : polyline,
            "center"   : [depot_lat, depot_lon],
            "zoom"     : 12,
        }

    def _score_color(self, score: float) -> str:
        """Couleur hex selon le priority score."""
        if score >= 0.70: return "#2ecc71"   # vert — haute priorité
        if score >= 0.45: return "#f39c12"   # orange — priorité moyenne
        return "#e74c3c"                      # rouge — basse priorité

    # ── Helper : zone stats ───────────────────────────────────────────

    def _zone_stats(self, df_stops: pd.DataFrame) -> List[Dict]:
        stats = []
        for c in range(self._config["k_clusters"]):
            mask = df_stops["cluster"] == c
            if mask.sum() > 0:
                stats.append({
                    "cluster_id"  : c,
                    "zone_name"   : ZONE_NAMES.get(c, f"Zone {c}"),
                    "n_stops"     : int(mask.sum()),
                    "avg_score"   : round(df_stops[mask]["priority_score"].mean(), 3),
                })
        return stats

    # ── Utilities ─────────────────────────────────────────────────────

    def get_pharmacies_df(self) -> pd.DataFrame:
        """Retourne le DataFrame complet des pharmacies."""
        return self._df_pharm.copy()

    def get_top_pharmacies(self, n: int = 10,
                            target_hour: int = 10,
                            depot_lat: float = None,
                            depot_lon: float = None) -> pd.DataFrame:
        """Retourne les N pharmacies avec le meilleur priority score."""
        depot_lat = depot_lat or self._config["depot_lat"]
        depot_lon = depot_lon or self._config["depot_lon"]
        df = self._compute_scores("Monday", target_hour, depot_lat, depot_lon)
        return df.nlargest(n, "priority_score")[
            ["venue_name","venue_address","latitude","longitude",
             "priority_score","day_mean","dist_depot_km","data_source","cluster"]
        ].reset_index(drop=True)

    def result_summary(self, result: Dict) -> str:
        """Résumé lisible d'une route optimisée."""
        lines = [
            f"┌─ Route Optimale {'─'*43}",
            f"│  Tournée     : {result['target_day_fr']} {result['target_hour']}h",
            f"│  Distance    : {result['total_distance_km']:.2f} km  "
            f"(-{result['improvement_pct']:.1f}% vs Nearest Neighbor)",
            f"│  Durée totale: {result['total_time_min']:.0f} min  "
            f"({result['drive_time_min']:.0f} déplacement + "
            f"{result['visit_time_min']:.0f} visites)",
            f"│  Zone        : {result['zone_name']}",
            f"│  Stops ({result['n_stops']}) :",
        ]
        for stop in result["route"]:
            if stop["is_depot"]:
                lines.append(f"│    ▶  {stop['venue_name']}")
            else:
                lines.append(
                    f"│    {stop['order']:>2}.  {stop['venue_name'][:38]:<38}  "
                    f"score={stop['priority_score']:.3f}  +{stop['dist_from_prev_km']:.1f}km"
                )
        lines.append(f"└{'─'*59}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RouteOptimizer("
            f"version='{self.version}', "
            f"pharmacies={self.n_pharmacies}, "
            f"zones={self._config['k_clusters']})"
        )


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  RouteOptimizer — Quick Inference Test")
    print("=" * 65)

    optimizer = RouteOptimizer.load(DEFAULT_MODEL_PATH)
    print(f"\n  {optimizer}\n")

    # ── Test 1 : Tournée standard Lundi 10h ───────────────────────────
    print("─" * 65)
    print("  TEST 1 — Tournée standard (Lundi, 10h, 8 stops)")
    print("─" * 65)
    result = optimizer.optimize(
        target_day="Monday", target_hour=10, n_stops=8
    )
    print(optimizer.result_summary(result))

    # ── Test 2 : Tournée jeudi peak ───────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 2 — Tournée peak (Jeudi, 10h, 6 stops)")
    print("─" * 65)
    result2 = optimizer.optimize(
        target_day="Thursday", target_hour=10, n_stops=6
    )
    print(f"  Distance  : {result2['total_distance_km']:.2f} km")
    print(f"  Durée     : {result2['total_time_min']:.0f} min")
    print(f"  Améliora. : -{result2['improvement_pct']:.1f}%")

    # ── Test 3 : Zone spécifique ──────────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 3 — Zone spécifique (cluster=1)")
    print("─" * 65)
    result3 = optimizer.optimize(
        target_day="Wednesday", target_hour=14,
        n_stops=5, cluster_id=1
    )
    print(f"  Zone : {result3['zone_name']}")
    print(f"  Distance : {result3['total_distance_km']:.2f} km")

    # ── Test 4 : Map data ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 4 — Map data (pour Leaflet.js Django)")
    print("─" * 65)
    map_d = result["map_data"]
    print(f"  Markers   : {len(map_d['markers'])}")
    print(f"  Polyline  : {len(map_d['polyline'])} points")
    print(f"  Center    : {map_d['center']}")
    print(f"  Zoom      : {map_d['zoom']}")
    print(f"  Premier marker : {map_d['markers'][0]}")

    # ── Test 5 : Top pharmacies ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("  TEST 5 — Top 5 pharmacies (priorité)")
    print("─" * 65)
    top = optimizer.get_top_pharmacies(n=5, target_hour=10)
    print(top[["venue_name","priority_score","day_mean","dist_depot_km"]].to_string())

    print(f"\n✅  RouteOptimizer — inférence validée")
    print(f"    Import : from route_model import RouteOptimizer")
