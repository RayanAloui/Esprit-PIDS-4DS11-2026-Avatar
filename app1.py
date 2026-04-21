import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# -------------------------
# CONFIG PAGE
# -------------------------
st.set_page_config(page_title="Optimisation Trajet", layout="wide")

# -------------------------
# STYLE (DESIGN PRO)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00ffcc;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Smart Route Optimization")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("datasets_final_24h_clean.xlsx")
    
    df = df.drop_duplicates(subset='client_id')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    df = df[
        (df['latitude'] >= 30) & (df['latitude'] <= 38) &
        (df['longitude'] >= 7) & (df['longitude'] <= 12)
    ]
    
    df = df[(df['latitude'] > 36.7) & (df['longitude'] > 10.0)]
    
    df['patients_per_month'] = pd.to_numeric(df['patients_per_month'], errors='coerce')
    
    mapping = {'Faible':1, 'Moyen':2, 'Élevé':3, 'Eleve':3}
    df['engagement_score'] = df['engagement_level'].map(mapping)
    
    return df

df = load_data()

# -------------------------
# FUNCTIONS
# -------------------------
def compute_distance(p1, p2):
    return geodesic(p1, p2).km

def get_activity(row, hour):
    return row.get(f'hour_{int(hour)}', 0)

def compute_traffic_score(activity):
    return 0 if activity == 0 else 1 / (activity + 1)

def compute_dynamic_score(df, user_location, start_hour):
    df = df.copy()
    scores = []
    
    for _, row in df.iterrows():
        dist = compute_distance(user_location, (row['latitude'], row['longitude']))
        distance_score = 1 / (dist + 0.001)
        
        activity = get_activity(row, start_hour)
        traffic_score = compute_traffic_score(activity)
        
        patients = row['patients_per_month'] if not pd.isna(row['patients_per_month']) else 0
        patients_norm = patients / df['patients_per_month'].max()
        
        engagement = row['engagement_score'] if not pd.isna(row['engagement_score']) else 0
        engagement_norm = engagement / df['engagement_score'].max()
        
        score = 0.4*distance_score + 0.3*traffic_score + 0.2*patients_norm + 0.1*engagement_norm
        scores.append(score)
    
    df['score'] = scores
    return df

def remove_duplicates(df):
    return df.sort_values(by='score', ascending=False).drop_duplicates('client_id')

def select_best(df, n):
    return df.sort_values(by='score', ascending=False).head(n)

def compute_route(df, user_location):
    points = df.copy()
    route = []
    current = user_location
    
    while len(points) > 0:
        points['dist'] = points.apply(
            lambda r: compute_distance(current, (r['latitude'], r['longitude'])),
            axis=1
        )
        nearest = points.loc[points['dist'].idxmin()]
        route.append(nearest)
        current = (nearest['latitude'], nearest['longitude'])
        points = points.drop(nearest.name)
    
    return pd.DataFrame(route)

def create_map(df, route, user_location):
    m = folium.Map(location=user_location, zoom_start=12)
    
    folium.Marker(user_location, popup="You", icon=folium.Icon(color='red')).add_to(m)
    
    cluster = MarkerCluster().add_to(m)
    for _, r in df.iterrows():
        folium.CircleMarker((r['latitude'], r['longitude']), radius=3, color='gray').add_to(cluster)
    
    coords = [user_location]
    
    for i, r in route.iterrows():
        c = (r['latitude'], r['longitude'])
        coords.append(c)
        
        folium.Marker(
            c,
            popup=f"{i+1} - {r['client_name']}",
            icon=folium.Icon(color='green' if r['score']>0.7 else 'blue')
        ).add_to(m)
    
    folium.PolyLine(coords, color="green").add_to(m)
    
    return m

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Paramètres")

start_hour = st.sidebar.slider("Heure", 0, 23, 9)
n_points = st.sidebar.slider("Nombre de visites", 1, 10, 5)

# -------------------------
# SESSION STATE
# -------------------------
if "route" not in st.session_state:
    st.session_state.route = None

if "user_location" not in st.session_state:
    st.session_state.user_location = (36.85, 10.20)

# -------------------------
# MAP CLICK
# -------------------------
st.subheader("📍 Choisissez votre position (clic sur map)")

map_click = folium.Map(location=st.session_state.user_location, zoom_start=12)
map_data = st_folium(map_click, height=400, width=700)

if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state.user_location = (lat, lon)

st.write("Position actuelle :", st.session_state.user_location)

# -------------------------
# BUTTON
# -------------------------
if st.button("🚀 Calculer trajet"):
    
    user_location = st.session_state.user_location
    
    df_scored = compute_dynamic_score(df, user_location, start_hour)
    df_unique = remove_duplicates(df_scored)
    best = select_best(df_unique, n_points)
    route = compute_route(best, user_location)
    
    st.session_state.route = route

# -------------------------
# DISPLAY
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    if st.session_state.route is not None:
        m = create_map(df, st.session_state.route, st.session_state.user_location)
        st_folium(m, height=500)

with col2:
    st.subheader("📊 Stats")
    
    if st.session_state.route is not None:
        route = st.session_state.route
        
        st.metric("Points visités", len(route))
        st.metric("Score moyen", round(route['score'].mean(),2))
        st.metric("Distance approx", round(route['dist'].sum(),2) if 'dist' in route else 0)