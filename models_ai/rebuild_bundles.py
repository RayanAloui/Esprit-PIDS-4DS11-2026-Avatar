"""
rebuild_bundles.py
==================
Script de reconstruction des bundles .pkl
A executer depuis le dossier models_ai/

Usage:
    cd models_ai
    python rebuild_bundles.py
"""
import sys
import os

# S'assurer qu'on est dans le bon dossier
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("=" * 60)
print("  REBUILD BUNDLES — ALIA Project")
print(f"  Dossier : {script_dir}")
print("=" * 60)

# ══════════════════════════════════════════════════════
# BUNDLE 1 — NLP Scoring
# ══════════════════════════════════════════════════════
print("\n[1/2] Reconstruction bundle NLP Scoring...")

try:
    import joblib
    import numpy as np
    import pandas as pd
    import re
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import f1_score, r2_score
    from scipy.sparse import hstack, csr_matrix

    # Importer la classe depuis son module
    from nlp_scoring_train_v2 import NLPFeatureExtractorV2

    SEED = 42
    np.random.seed(SEED)

    # Chercher le dataset
    data_path = None
    for candidate in [
        'conversation_transcripts_v2.csv',
        '../conversation_transcripts_v2.csv',
        '../../conversation_transcripts_v2.csv',
    ]:
        if os.path.exists(candidate):
            data_path = candidate
            break

    if data_path is None:
        print("  ERREUR: conversation_transcripts_v2.csv introuvable")
        print("  Copiez ce fichier dans le dossier models_ai/")
        raise FileNotFoundError("conversation_transcripts_v2.csv")

    print(f"  Dataset : {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['input_text'] = df['objection_text'] + ' [SEP] ' + df['rep_response']
    print(f"  Shape   : {df.shape}")

    # Feature extraction
    extractor = NLPFeatureExtractorV2()
    X_ling = np.array([
        list(extractor.extract(r['objection_text'], r['rep_response']).values())
        for _, r in df.iterrows()
    ])

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=8000,
                            min_df=2, sublinear_tf=True)
    X_tf   = tfidf.fit_transform(df['input_text'])
    sc     = StandardScaler()
    X_l    = csr_matrix(sc.fit_transform(X_ling))
    X      = hstack([X_tf, X_l])

    le1 = LabelEncoder(); le3 = LabelEncoder()
    le4 = LabelEncoder(); le5 = LabelEncoder()
    y1  = le1.fit_transform(df['response_quality'])
    y2  = df[['scientific_accuracy_score',
               'communication_clarity_score',
               'objection_handling_score']].values
    y3  = le3.fit_transform(df['client_sentiment'])
    y4  = le4.fit_transform(df['niveau_alia'])
    y5  = le5.fit_transform(df['visit_format'])
    y6  = df['conformite_flag'].values

    Xtr, Xte, i_tr, i_te = train_test_split(
        X, np.arange(len(df)), test_size=0.2,
        random_state=SEED, stratify=y1)

    print("  Entraînement T1 (SVM)...")
    m1 = SVC(kernel='rbf', C=5.0, gamma='scale',
             probability=True, random_state=SEED)
    m1.fit(Xtr, y1[i_tr])

    print("  Entraînement T2 (Ridge)...")
    m2 = MultiOutputRegressor(Ridge(1.0))
    m2.fit(Xtr, y2[i_tr])

    print("  Entraînement T3 (LogReg)...")
    m3 = LogisticRegression(max_iter=2000, C=0.1, random_state=SEED)
    m3.fit(Xtr, y3[i_tr])

    print("  Entraînement T4 (SVM niveau ALIA)...")
    m4 = SVC(kernel='rbf', C=10.0, gamma='scale',
             probability=True, random_state=SEED)
    m4.fit(Xtr, y4[i_tr])

    print("  Entraînement T5 (format visite)...")
    m5 = LogisticRegression(max_iter=2000, random_state=SEED)
    m5.fit(Xtr, y5[i_tr])

    print("  Entraînement T6 (conformité)...")
    m6 = LogisticRegression(max_iter=2000, random_state=SEED)
    m6.fit(Xtr, y6[i_tr])

    # Métriques
    f1_t1 = f1_score(y1[i_te], m1.predict(Xte), average='macro')
    p2    = m2.predict(Xte)
    w     = np.array([0.25, 0.30, 0.45])
    r2_t2 = r2_score((y2[i_te]*w).sum(1), (p2*w).sum(1))
    f1_t4 = f1_score(y4[i_te], m4.predict(Xte), average='macro')

    print(f"  T1 F1={f1_t1:.4f} | T2 R²={r2_t2:.4f} | T4 F1={f1_t4:.4f}")

    CONFIG = {
        'score_weights'    : [0.25, 0.30, 0.45],
        'seuil_competence' : 7.0,
        'tfidf'            : {'ngram_range':(1,2),'max_features':8000,'min_df':2,'sublinear_tf':True},
        'seuils_alia'      : {'Débutant':7.0,'Junior':8.0,'Confirmé':9.0,'Expert':10.0},
        'weight_affluence' : 0.40,
        'weight_peak'      : 0.35,
        'weight_geo'       : 0.25,
        'avg_speed_kmh'    : 35.0,
        'visit_duration_min': 5.0,
    }

    bundle_nlp = {
        'tfidf'      : tfidf,
        'scaler'     : sc,
        'extractor'  : extractor,
        'encoders'   : {'t1':le1,'t3':le3,'t4':le4,'t5':le5},
        'models'     : {'t1':m1,'t2':m2,'t3':m3,'t4':m4,'t5':m5,'t6':m6},
        'config'     : CONFIG,
        'class_names': {
            't1': list(le1.classes_),
            't3': list(le3.classes_),
            't4': list(le4.classes_),
            't5': list(le5.classes_),
        },
        'seuils_alia': CONFIG['seuils_alia'],
        'version'    : '2.0.0',
    }

    joblib.dump(bundle_nlp, 'nlp_scoring_bundle_v2.pkl')
    size = os.path.getsize('nlp_scoring_bundle_v2.pkl') / 1024
    print(f"  ✅  nlp_scoring_bundle_v2.pkl sauvegardé ({size:.1f} KB)")

except Exception as e:
    print(f"  ❌  ERREUR NLP : {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════
# BUNDLE 2 — Route Optimizer
# ══════════════════════════════════════════════════════
print("\n[2/2] Reconstruction bundle Route Optimizer...")

try:
    import joblib
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    from datetime import datetime

    # Chercher le dataset
    data_path = None
    for candidate in [
        'pharmacies_foot_traffic.csv',
        '../pharmacies_foot_traffic.csv',
        '../../pharmacies_foot_traffic.csv',
    ]:
        if os.path.exists(candidate):
            data_path = candidate
            break

    if data_path is None:
        # Essayer avec le fichier déjà traité
        for candidate in [
            'pharmacies_grand_tunis.csv',
            '../pharmacies_grand_tunis.csv',
        ]:
            if os.path.exists(candidate):
                data_path = candidate
                break

    if data_path is None:
        print("  ERREUR: pharmacies_foot_traffic.csv introuvable")
        raise FileNotFoundError("pharmacies_foot_traffic.csv")

    print(f"  Dataset : {data_path}")

    HOUR_COLS = [f'hour_{i}' for i in range(24)]

    def haversine_matrix(lats1, lons1, lats2, lons2):
        R = 6371.0
        lat1,lon1 = np.radians(lats1), np.radians(lons1)
        lat2,lon2 = np.radians(lats2), np.radians(lons2)
        dlat = lat2[:,None]-lat1[None,:]; dlon = lon2[:,None]-lon1[None,:]
        a = np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2[:,None])*np.sin(dlon/2)**2
        return R*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

    # Charger et préparer les données
    if 'foot_traffic' in data_path or 'pharmacies_foot' in data_path:
        df_raw = pd.read_csv(data_path, encoding='utf-8-sig')
        df_t   = df_raw[df_raw['latitude'] > 35].copy()
        pharm  = df_t[['venue_id','venue_name','venue_address',
                        'latitude','longitude','forecast_available']
                      ].drop_duplicates().reset_index(drop=True)
        agg    = (df_t[df_t['forecast_available']==True]
                  .groupby('venue_id')[['day_mean','day_max']+HOUR_COLS]
                  .mean().round(2).reset_index())
        df     = pharm.merge(agg, on='venue_id', how='left')
    else:
        df = pd.read_csv(data_path, encoding='utf-8-sig')

    print(f"  Shape   : {df.shape}")

    # Imputation k-NN
    known   = df[df['day_mean'].notna()].copy()
    unknown = df[df['day_mean'].isna()].copy()

    if len(unknown) > 0:
        dist_mat = haversine_matrix(
            known['latitude'].values, known['longitude'].values,
            unknown['latitude'].values, unknown['longitude'].values)
        cols_imp = ['day_mean','day_max'] + HOUR_COLS
        for idx_u, row_dist in enumerate(dist_mat):
            top_k   = np.argsort(row_dist)[:3]
            dists   = row_dist[top_k]+1e-6
            weights = (1/dists)/(1/dists).sum()
            for col in cols_imp:
                if col in unknown.columns:
                    unknown.iloc[idx_u, df.columns.get_loc(col)] = \
                        (known.iloc[top_k][col].values * weights).sum()

    df_full = pd.concat([known, unknown], ignore_index=True)
    df_full['data_source'] = np.where(
        df_full['forecast_available'], 'réel', 'imputé')

    # Priority score
    DEPOT_LAT, DEPOT_LON = 36.8190, 10.1660
    df_full['affluence_norm'] = MinMaxScaler().fit_transform(df_full[['day_mean']])
    hcol = 'hour_10'
    if hcol in df_full.columns:
        df_full['peak_score'] = MinMaxScaler().fit_transform(df_full[[hcol]])
    else:
        df_full['peak_score'] = df_full['affluence_norm']
    dists = haversine_matrix(
        np.array([DEPOT_LAT]), np.array([DEPOT_LON]),
        df_full['latitude'].values, df_full['longitude'].values).flatten()
    df_full['dist_depot_km'] = dists.round(2)
    df_full['geo_score']     = (1.0-dists/(dists.max()+1e-6)).round(4)
    df_full['priority_score'] = (
        0.40*df_full['affluence_norm'] +
        0.35*df_full['peak_score']     +
        0.25*df_full['geo_score']).round(4)

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_full['cluster'] = kmeans.fit_predict(
        df_full[['latitude','longitude']].values)

    print(f"  Pharmacies : {len(df_full)} | Clusters : 4")
    print(f"  Score range : {df_full['priority_score'].min():.3f}"
          f" – {df_full['priority_score'].max():.3f}")

    CONFIG_ROUTE = {
        'depot_lat'         : DEPOT_LAT,
        'depot_lon'         : DEPOT_LON,
        'depot_name'        : 'VITAL SA — Siège (Tunis)',
        'max_stops'         : 8,
        'min_stops'         : 4,
        'k_clusters'        : 4,
        'knn_k'             : 3,
        'weight_affluence'  : 0.40,
        'weight_peak'       : 0.35,
        'weight_geo'        : 0.25,
        'avg_speed_kmh'     : 35.0,
        'visit_duration_min': 5.0,
    }

    bundle_route = {
        'kmeans'       : kmeans,
        'df_pharmacies': df_full,
        'config'       : CONFIG_ROUTE,
        'trained_at'   : datetime.now().isoformat(),
        'version'      : '1.0.0',
    }

    joblib.dump(bundle_route, 'route_model.pkl')
    size = os.path.getsize('route_model.pkl') / 1024
    print(f"  ✅  route_model.pkl sauvegardé ({size:.1f} KB)")

except Exception as e:
    print(f"  ❌  ERREUR Route : {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RÉSUMÉ")
print("=" * 60)
for fname in ['nlp_scoring_bundle_v2.pkl', 'route_model.pkl',
              'lstm_body_language_v2.pkl']:
    path = fname
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✅  {fname:<40}  {size:>7.1f} KB")
    else:
        print(f"  ❌  {fname}  — MANQUANT")

print()
print("  Lancez maintenant depuis alia_django/ :")
print("       python manage.py runserver")
