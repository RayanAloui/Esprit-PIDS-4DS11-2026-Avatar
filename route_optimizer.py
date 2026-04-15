"""
route_optimizer.py
==================
DSO 4 — ALIA Avatar Project | TDSP Phase 3 : Model Development
---------------------------------------------------------------
Pipeline de production pour l'optimisation des tournées de délégués
médicaux VITAL SA sur le Grand Tunis.

Pipeline complet :
    DS6 pharmacies_foot_traffic.csv
        → Nettoyage + imputation spatiale (k-NN, k=3)
            → Priority Score (affluence + peak + géo)
                → Clustering K-Means (4 zones Grand Tunis)
                    → TSP Nearest Neighbor (solution initiale)
                        → 2-Opt Local Search (amélioration)
                            → models/route_model.pkl

Usage :
    python route_optimizer.py                         # pipeline standard
    python route_optimizer.py --day Thursday --hour 10
    python route_optimizer.py --stops 6 --cluster 1
    python route_optimizer.py --depot-lat 36.85 --depot-lon 10.32
    python route_optimizer.py --eval-only

Output :
    models/route_model.pkl          ← bundle complet
    models/route_training_report.json
    outputs/route_optimale.json     ← résultat tournée
    models/route_optimizer.log

Author  : CYBER SHADE — ALIA Project
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Optional, Tuple

# ── Scientific stack ──────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster      import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    "data_path"    : "pharmacies_foot_traffic.csv",
    "models_dir"   : "models",
    "outputs_dir"  : "outputs",
    "lat_filter"   : 35.0,       # Grand Tunis uniquement (lat > 35°N)

    # ── Dépôt par défaut (VITAL SA — Tunis) ───────────────────────────
    "depot_lat"    : 36.8190,
    "depot_lon"    : 10.1660,
    "depot_name"   : "VITAL SA — Siège (Tunis)",

    # ── Tournée ───────────────────────────────────────────────────────
    "max_stops"    : 8,          # stops max par tournée
    "min_stops"    : 4,
    "default_day"  : "Monday",
    "default_hour" : 10,

    # ── Algorithme ────────────────────────────────────────────────────
    "knn_k"        : 3,          # voisins pour imputation spatiale
    "k_clusters"   : 4,          # zones Grand Tunis
    "two_opt_max_iter": 500,     # itérations max 2-Opt

    # ── Priority Score weights ─────────────────────────────────────────
    "weight_affluence": 0.40,    # day_mean normalisé
    "weight_peak"     : 0.35,    # affluence heure cible
    "weight_geo"      : 0.25,    # proximité dépôt

    # ── Vitesse déplacement estimée ───────────────────────────────────
    "avg_speed_kmh"  : 35.0,    # Grand Tunis en voiture
    "visit_duration_min": 5.0,  # durée visite par pharmacie
}

HOUR_COLS  = [f"hour_{i}" for i in range(24)]
DAYS_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

os.makedirs(CONFIG["models_dir"],  exist_ok=True)
os.makedirs(CONFIG["outputs_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(CONFIG["models_dir"]) / "route_optimizer.log", mode="w"
        ),
    ],
)
log = logging.getLogger("RouteOptimizer")


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS GÉOSPATIALES
# ══════════════════════════════════════════════════════════════════════

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Distance haversine (km) entre deux points GPS."""
    R    = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat1)) *
            np.cos(np.radians(lat2)) *
            np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_matrix(lats1: np.ndarray, lons1: np.ndarray,
                     lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """
    Matrice de distances haversine (km).

    Returns:
        (len(lats2), len(lats1)) matrix
    """
    R    = 6371.0
    lat1 = np.radians(lats1); lon1 = np.radians(lons1)
    lat2 = np.radians(lats2); lon2 = np.radians(lons2)
    dlat = lat2[:, None] - lat1[None, :]
    dlon = lon2[:, None] - lon1[None, :]
    a    = (np.sin(dlat/2)**2 +
            np.cos(lat1) * np.cos(lat2[:, None]) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def route_distance(route: List[int],
                   lats: np.ndarray, lons: np.ndarray) -> float:
    """Distance totale d'une route (km) — dépôt inclus."""
    return sum(
        haversine_distance(lats[route[i]], lons[route[i]],
                           lats[route[i+1]], lons[route[i+1]])
        for i in range(len(route) - 1)
    )


# ══════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════

class PharmacyDataLoader:
    """Charge, nettoie et impute les données pharmacies DS6."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        path = Path(self.cfg["data_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {path}\n"
                "Placez pharmacies_foot_traffic.csv dans le répertoire."
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
        log.info(f"Dataset brut chargé : {df.shape}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre le Grand Tunis et agrège le foot traffic."""
        # Filtre géographique Grand Tunis
        df_tunis = df[df["latitude"] > self.cfg["lat_filter"]].copy()

        # Pharmacies uniques
        pharm = df_tunis[
            ["venue_id","venue_name","venue_address",
             "latitude","longitude","forecast_available"]
        ].drop_duplicates().reset_index(drop=True)

        # Agréger foot traffic (moyenne tous jours)
        agg = (df_tunis[df_tunis["forecast_available"]]
               .groupby("venue_id")[["day_mean","day_max"] + HOUR_COLS]
               .mean().round(2).reset_index())

        df_clean = pharm.merge(agg, on="venue_id", how="left")

        log.info(f"Grand Tunis        : {len(df_clean)} pharmacies")
        log.info(f"  Données réelles  : {df_clean['day_mean'].notna().sum()}")
        log.info(f"  À imputer (k-NN) : {df_clean['day_mean'].isna().sum()}")

        return df_clean

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputation spatiale k-NN pour les pharmacies sans données.
        Chaque pharmacie inconnue reçoit la moyenne pondérée (1/distance)
        de ses k=3 voisins les plus proches ayant des données.
        """
        k = self.cfg["knn_k"]
        cols_to_impute = ["day_mean","day_max"] + HOUR_COLS

        known_mask   = df["day_mean"].notna()
        known        = df[known_mask].copy()
        unknown      = df[~known_mask].copy()

        if len(unknown) == 0:
            log.info("Imputation : aucune valeur manquante.")
            df["data_source"] = "réel"
            return df

        dist_mat = haversine_matrix(
            known["latitude"].values,   known["longitude"].values,
            unknown["latitude"].values, unknown["longitude"].values,
        )  # (n_unknown, n_known)

        for idx_u, row_dist in enumerate(dist_mat):
            top_k   = np.argsort(row_dist)[:k]
            dists   = row_dist[top_k] + 1e-6
            weights = 1.0 / dists
            weights /= weights.sum()
            for col in cols_to_impute:
                unknown.iloc[idx_u, df.columns.get_loc(col)] = (
                    (known.iloc[top_k][col].values * weights).sum()
                )

        df_full = pd.concat([known, unknown], ignore_index=True)
        df_full["data_source"] = np.where(
            df_full["forecast_available"], "réel", "imputé"
        )

        log.info(f"Imputation k-NN (k={k}) : OK")
        log.info(f"  NaN restants : {df_full['day_mean'].isna().sum()}")
        return df_full


# ══════════════════════════════════════════════════════════════════════
# PRIORITY SCORER
# ══════════════════════════════════════════════════════════════════════

class PriorityScorer:
    """
    Calcule le Priority Score de chaque pharmacie selon la formule VITAL SA :

        priority = w_a × affluence_norm
                 + w_p × peak_score
                 + w_g × geo_score

    Poids : 0.40 / 0.35 / 0.25 (configurables)
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def compute(self, df: pd.DataFrame,
                target_day : str = "Monday",
                target_hour: int = 10,
                depot_lat  : float = None,
                depot_lon  : float = None) -> pd.DataFrame:
        """
        Args:
            df          : DataFrame pharmacies (avec foot traffic imputé)
            target_day  : jour cible de la tournée
            target_hour : heure cible (0–23)
            depot_lat/lon: point de départ (dépôt délégué)

        Returns:
            df avec colonnes priority_score, affluence_norm, peak_score, geo_score
        """
        depot_lat = depot_lat or self.cfg["depot_lat"]
        depot_lon = depot_lon or self.cfg["depot_lon"]
        df        = df.copy()

        # 1. Affluence normalisée (day_mean)
        df["affluence_norm"] = MinMaxScaler().fit_transform(df[["day_mean"]])

        # 2. Peak score : affluence à l'heure cible
        hour_col = f"hour_{target_hour}"
        if hour_col in df.columns:
            df["peak_score"] = MinMaxScaler().fit_transform(df[[hour_col]])
        else:
            df["peak_score"] = df["affluence_norm"]

        # 3. Geo score : inverse de la distance au dépôt
        dists = haversine_matrix(
            np.array([depot_lat]), np.array([depot_lon]),
            df["latitude"].values,  df["longitude"].values
        ).flatten()
        df["dist_depot_km"] = dists.round(2)
        max_dist = dists.max() + 1e-6
        df["geo_score"] = (1.0 - dists / max_dist).round(4)

        # 4. Score composite
        wa = self.cfg["weight_affluence"]
        wp = self.cfg["weight_peak"]
        wg = self.cfg["weight_geo"]
        df["priority_score"] = (
            wa * df["affluence_norm"] +
            wp * df["peak_score"]     +
            wg * df["geo_score"]
        ).round(4)

        log.info(f"Priority Scores calculés ({target_day} {target_hour}h) :")
        log.info(f"  Range : {df['priority_score'].min():.3f} – "
                 f"{df['priority_score'].max():.3f}")
        log.info(f"  Moyenne : {df['priority_score'].mean():.3f}")

        return df


# ══════════════════════════════════════════════════════════════════════
# CLUSTER MANAGER
# ══════════════════════════════════════════════════════════════════════

class ClusterManager:
    """K-Means clustering géographique — 4 zones Grand Tunis."""

    def __init__(self, cfg: Dict):
        self.cfg    = cfg
        self.kmeans : Optional[KMeans] = None
        self.zone_names = {
            0: "Zone A — Nord-Est (La Marsa / Carthage)",
            1: "Zone B — Centre (Tunis Médina / Belvédère)",
            2: "Zone C — Centre-Ouest (Bardo / Manouba)",
            3: "Zone D — Sud (Ben Arous / Mégrine)",
        }

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entraîne le K-Means et assigne les clusters."""
        k = self.cfg["k_clusters"]
        coords = df[["latitude","longitude"]].values

        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["cluster"] = self.kmeans.fit_predict(coords)

        log.info(f"Clustering K-Means (k={k}) :")
        for c in range(k):
            n = (df["cluster"] == c).sum()
            log.info(f"  {self.zone_names.get(c, f'Zone {c}')} : {n} pharmacies")

        return df

    def predict(self, lat: float, lon: float) -> int:
        """Prédit la zone d'un nouveau point."""
        if self.kmeans is None:
            raise RuntimeError("ClusterManager non entraîné — appelez fit() d'abord.")
        return int(self.kmeans.predict([[lat, lon]])[0])

    def filter_cluster(self, df: pd.DataFrame,
                       cluster_id: int) -> pd.DataFrame:
        """Retourne uniquement les pharmacies d'une zone."""
        return df[df["cluster"] == cluster_id]


# ══════════════════════════════════════════════════════════════════════
# TSP SOLVER
# ══════════════════════════════════════════════════════════════════════

class TSPSolver:
    """
    Résolution du TSP par heuristiques :
    1. Nearest Neighbor (solution initiale)
    2. 2-Opt Local Search (amélioration)
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def nearest_neighbor(self, lats: np.ndarray, lons: np.ndarray,
                         start_idx: int = 0) -> Tuple[List[int], float]:
        """
        Heuristique Nearest Neighbor.

        Args:
            lats, lons  : coordonnées (index 0 = dépôt)
            start_idx   : index de départ

        Returns:
            route : liste d'indices (aller-retour)
            dist  : distance totale (km)
        """
        n       = len(lats)
        visited = [False] * n
        route   = [start_idx]
        visited[start_idx] = True
        current = start_idx

        for _ in range(n - 1):
            best_dist = np.inf
            best_next = -1
            for j in range(n):
                if not visited[j]:
                    d = haversine_distance(lats[current], lons[current],
                                           lats[j], lons[j])
                    if d < best_dist:
                        best_dist = d
                        best_next = j
            route.append(best_next)
            visited[best_next] = True
            current = best_next

        route.append(start_idx)  # retour dépôt
        return route, route_distance(route, lats, lons)

    def two_opt(self, route: List[int],
                lats: np.ndarray, lons: np.ndarray,
                max_iter: int = None) -> Tuple[List[int], float, List[float]]:
        """
        2-Opt Local Search — améliore la route par échanges d'arêtes.

        Returns:
            best_route : route améliorée
            best_dist  : distance totale (km)
            history    : distances à chaque amélioration
        """
        max_iter   = max_iter or self.cfg["two_opt_max_iter"]
        best_route = route[:]
        best_dist  = route_distance(best_route, lats, lons)
        history    = [best_dist]
        improved   = True
        iteration  = 0

        while improved and iteration < max_iter:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = (best_route[:i] +
                                 best_route[i:j+1][::-1] +
                                 best_route[j+1:])
                    new_dist  = route_distance(new_route, lats, lons)
                    if new_dist < best_dist - 1e-6:
                        best_route = new_route
                        best_dist  = new_dist
                        history.append(best_dist)
                        improved   = True
            iteration += 1

        return best_route, best_dist, history

    def solve(self, df_stops: pd.DataFrame,
              depot_lat: float, depot_lon: float,
              depot_name: str) -> Dict:
        """
        Résolution complète pour un ensemble de stops.

        Args:
            df_stops   : DataFrame des pharmacies à visiter
            depot_lat/lon/name : point de départ/arrivée

        Returns:
            dict complet avec route, distances, durées
        """
        # Construire le tableau de points (dépôt en index 0)
        depot_row = pd.DataFrame([{
            "venue_name"    : depot_name,
            "latitude"      : depot_lat,
            "longitude"     : depot_lon,
            "priority_score": 1.0,
            "cluster"       : -1,
            "data_source"   : "dépôt",
        }])
        df_route = pd.concat([depot_row, df_stops], ignore_index=True)

        lats  = df_route["latitude"].values
        lons  = df_route["longitude"].values

        # 1. Nearest Neighbor
        t0 = time.time()
        route_nn, dist_nn = self.nearest_neighbor(lats, lons, start_idx=0)
        time_nn = time.time() - t0

        # 2. 2-Opt
        t0 = time.time()
        route_2opt, dist_2opt, history = self.two_opt(route_nn, lats, lons)
        time_2opt = time.time() - t0

        improvement = (dist_nn - dist_2opt) / dist_nn * 100

        log.info(f"  Nearest Neighbor : {dist_nn:.2f} km ({time_nn*1000:.1f}ms)")
        log.info(f"  2-Opt optimisé   : {dist_2opt:.2f} km ({time_2opt*1000:.1f}ms)")
        log.info(f"  Amélioration     : -{improvement:.1f}%")

        # Construire le résultat détaillé
        drive_time  = dist_2opt / self.cfg["avg_speed_kmh"]
        visit_time  = len(df_stops) * self.cfg["visit_duration_min"] / 60
        total_time  = drive_time + visit_time

        stops_detail = []
        for i, idx in enumerate(route_2opt):
            row = df_route.iloc[idx]
            d_from_prev = 0.0
            if i > 0:
                prev_idx = route_2opt[i-1]
                d_from_prev = haversine_distance(
                    lats[prev_idx], lons[prev_idx], lats[idx], lons[idx]
                )
            stops_detail.append({
                "order"         : i,
                "is_depot"      : idx == 0,
                "venue_name"    : str(row["venue_name"]),
                "address"       : str(row.get("venue_address", "")),
                "latitude"      : float(lats[idx]),
                "longitude"     : float(lons[idx]),
                "priority_score": float(row.get("priority_score", 0)),
                "cluster"       : int(row.get("cluster", -1)),
                "data_source"   : str(row.get("data_source", "—")),
                "dist_from_prev": round(d_from_prev, 2),
            })

        return {
            "route"            : stops_detail,
            "total_distance_km": round(dist_2opt, 2),
            "nn_distance_km"   : round(dist_nn, 2),
            "improvement_pct"  : round(improvement, 1),
            "n_stops"          : len(df_stops),
            "drive_time_min"   : round(drive_time * 60, 1),
            "visit_time_min"   : round(visit_time * 60, 1),
            "total_time_min"   : round(total_time * 60, 1),
            "history_2opt"     : [round(d, 2) for d in history],
            "n_improvements"   : len(history) - 1,
        }


# ══════════════════════════════════════════════════════════════════════
# MAIN TRAINER
# ══════════════════════════════════════════════════════════════════════

class RouteOptimizerTrainer:
    """Orchestre le pipeline complet d'optimisation."""

    def __init__(self, cfg: Dict):
        self.cfg     = cfg
        self.loader  = PharmacyDataLoader(cfg)
        self.scorer  = PriorityScorer(cfg)
        self.cluster = ClusterManager(cfg)
        self.solver  = TSPSolver(cfg)
        self.report  = {}

    def run(self, target_day: str = None, target_hour: int = None,
            n_stops: int = None, cluster_id: Optional[int] = None,
            depot_lat: float = None, depot_lon: float = None) -> Dict:
        """
        Pipeline complet :
        1. Chargement + nettoyage
        2. Imputation k-NN
        3. Priority Score
        4. Clustering K-Means
        5. Sélection des stops
        6. TSP Nearest Neighbor + 2-Opt
        7. Sauvegarde

        Returns:
            dict complet avec route optimisée
        """
        target_day  = target_day  or self.cfg["default_day"]
        target_hour = target_hour or self.cfg["default_hour"]
        n_stops     = n_stops     or self.cfg["max_stops"]
        depot_lat   = depot_lat   or self.cfg["depot_lat"]
        depot_lon   = depot_lon   or self.cfg["depot_lon"]

        log.info("=" * 70)
        log.info("  ALIA — Route Optimizer Pipeline V1")
        log.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"  Tournée  : {target_day} {target_hour}h | {n_stops} stops")
        log.info(f"  Dépôt    : ({depot_lat:.4f}, {depot_lon:.4f})")
        if cluster_id is not None:
            log.info(f"  Zone     : Cluster {cluster_id}")
        log.info("=" * 70)

        # ── 1–2. Chargement + imputation ──────────────────────────────
        df_raw   = self.loader.load()
        df_clean = self.loader.clean(df_raw)
        df_full  = self.loader.impute(df_clean)

        # ── 3. Priority Score ─────────────────────────────────────────
        df_scored = self.scorer.compute(
            df_full, target_day=target_day,
            target_hour=target_hour,
            depot_lat=depot_lat, depot_lon=depot_lon
        )

        # ── 4. Clustering ─────────────────────────────────────────────
        df_clustered = self.cluster.fit(df_scored)

        # ── 5. Sélection des stops ────────────────────────────────────
        if cluster_id is not None:
            df_candidates = self.cluster.filter_cluster(df_clustered, cluster_id)
            log.info(f"Zone {cluster_id} : {len(df_candidates)} candidats")
        else:
            df_candidates = df_clustered.copy()

        df_stops = df_candidates.nlargest(n_stops, "priority_score").copy()
        log.info(f"\n  Stops sélectionnés ({n_stops}) :")
        for _, r in df_stops.iterrows():
            log.info(f"    {r['priority_score']:.3f}  {r['venue_name'][:40]}")

        # ── 6. TSP ───────────────────────────────────────────────────
        log.info(f"\n  Optimisation TSP :")
        result = self.solver.solve(
            df_stops, depot_lat, depot_lon,
            self.cfg["depot_name"]
        )
        result.update({
            "target_day"  : target_day,
            "target_hour" : target_hour,
            "depot"       : {
                "name": self.cfg["depot_name"],
                "lat" : depot_lat, "lon": depot_lon,
            },
            "cluster_id"  : cluster_id,
            "computed_at" : datetime.now().isoformat(),
        })

        # ── 7. Affichage récap ────────────────────────────────────────
        log.info("\n" + "=" * 70)
        log.info(f"  ROUTE OPTIMALE — {target_day} {target_hour}h")
        log.info("=" * 70)
        for stop in result["route"]:
            tag = "▶ Dépôt" if stop["is_depot"] else f"  {stop['order']:>2}"
            log.info(f"  {tag}  {stop['venue_name'][:40]:<40}  "
                     f"+{stop['dist_from_prev']:.1f}km")
        log.info(f"\n  Distance totale : {result['total_distance_km']:.2f} km")
        log.info(f"  Durée totale    : {result['total_time_min']:.0f} min")
        log.info(f"  Amélioration    : -{result['improvement_pct']:.1f}% vs NN")
        log.info("=" * 70)

        # ── Sauvegarde ────────────────────────────────────────────────
        self._save(df_clustered, result)
        return result

    def _save(self, df_clustered: pd.DataFrame, result: Dict):
        mdir = Path(self.cfg["models_dir"])
        odir = Path(self.cfg["outputs_dir"])

        # Bundle modèle
        bundle = {
            "kmeans"          : self.cluster.kmeans,
            "df_pharmacies"   : df_clustered,
            "config"          : self.cfg,
            "scorer"          : self.scorer,
            "cluster_manager" : self.cluster,
            "solver"          : self.solver,
            "trained_at"      : datetime.now().isoformat(),
            "version"         : "1.0.0",
        }
        joblib.dump(bundle, mdir / "route_model.pkl")
        log.info(f"\n  ✅  Bundle sauvegardé → {mdir}/route_model.pkl")

        # Résultat tournée JSON
        with open(odir / "route_optimale.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"  ✅  Route sauvegardée → {odir}/route_optimale.json")

        # Rapport
        report = {
            "total_distance_km": result["total_distance_km"],
            "improvement_pct"  : result["improvement_pct"],
            "n_stops"          : result["n_stops"],
            "total_time_min"   : result["total_time_min"],
            "target_day"       : result["target_day"],
            "target_hour"      : result["target_hour"],
            "trained_at"       : datetime.now().isoformat(),
            "version"          : "1.0.0",
        }
        with open(mdir / "route_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Artifacts summary
        log.info("\n  Artifacts :")
        for p in sorted(mdir.iterdir()):
            if p.suffix in [".pkl", ".json", ".log"]:
                log.info(f"    {p.name:<45}  {p.stat().st_size/1024:>7.1f} KB")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ALIA Route Optimizer — Tournées délégués VITAL SA"
    )
    p.add_argument("--data",       default=CONFIG["data_path"],  help="Chemin dataset")
    p.add_argument("--day",        default=CONFIG["default_day"], help="Jour cible")
    p.add_argument("--hour",       type=int, default=CONFIG["default_hour"], help="Heure cible")
    p.add_argument("--stops",      type=int, default=CONFIG["max_stops"],    help="Nb stops max")
    p.add_argument("--cluster",    type=int, default=None, help="Zone (0-3), None=toutes zones")
    p.add_argument("--depot-lat",  type=float, default=CONFIG["depot_lat"])
    p.add_argument("--depot-lon",  type=float, default=CONFIG["depot_lon"])
    p.add_argument("--models-dir", default=CONFIG["models_dir"])
    p.add_argument("--eval-only",  action="store_true",
                   help="Charge le bundle existant et affiche la route")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["data_path"]  = args.data
    CONFIG["models_dir"] = args.models_dir
    CONFIG["depot_lat"]  = args.depot_lat
    CONFIG["depot_lon"]  = args.depot_lon
    os.makedirs(CONFIG["models_dir"],  exist_ok=True)
    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)

    if args.eval_only:
        bundle = joblib.load(Path(args.models_dir) / "route_model.pkl")
        log.info("Eval-only — bundle chargé.")
        log.info(f"  Pharmacies : {len(bundle['df_pharmacies'])}")
        log.info(f"  Version    : {bundle.get('version','?')}")
    else:
        trainer = RouteOptimizerTrainer(CONFIG)
        result  = trainer.run(
            target_day  = args.day,
            target_hour = args.hour,
            n_stops     = args.stops,
            cluster_id  = args.cluster,
            depot_lat   = args.depot_lat,
            depot_lon   = args.depot_lon,
        )
        log.info(f"\n✅  Pipeline complet — {result['total_distance_km']} km optimisés")
