import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="DSO Optimizer Pro", layout="wide")

st.title("🚀 DSO Optimizer Pro")

# ===============================
# SESSION STATE 📍
# ===============================
if "user_location" not in st.session_state:
    st.session_state["user_location"] = (36.8065, 10.1815)

if "temp_location" not in st.session_state:
    st.session_state["temp_location"] = None

# ===============================
# LOAD DATA (CACHE)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_fixed_features.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Paramètres")

lat = st.sidebar.number_input("Latitude", value=st.session_state["user_location"][0])
lon = st.sidebar.number_input("Longitude", value=st.session_state["user_location"][1])

st.session_state["user_location"] = (lat, lon)
user_location = st.session_state["user_location"]

day = st.sidebar.selectbox("Jour", sorted(df["day_text"].dropna().unique()))
hour = st.sidebar.selectbox("Heure", [f"hour_{i}" for i in range(24)])
n_pharma = st.sidebar.slider("Nb pharmacies auto", 3, 15, 8)

# ===============================
# FILTER
# ===============================
df_day = df[df["day_text"] == day].copy()
df_day["traffic"] = df_day[hour]
df_day = df_day[df_day["traffic"] > 0]

if df_day.empty:
    st.warning("Aucune pharmacie ouverte")
    st.stop()

# ===============================
# REGION DETECTION
# ===============================
possible_cols = ["governorate", "city", "region"]
region_col = next((c for c in possible_cols if c in df_day.columns), None)

if region_col is None:
    df_day["region"] = "Unknown"
    region_col = "region"

# ===============================
# REGION → PHARMACIES
# ===============================
st.subheader("🏙️ Sélection manuelle")

regions = sorted(df_day[region_col].dropna().unique())
selected_region = st.selectbox("Choisir région", regions)

df_region = df_day[df_day[region_col] == selected_region]

selected_pharmacies = st.multiselect(
    "Choisir pharmacies (optionnel)",
    df_region["name"]
)

# ===============================
# DISTANCE
# ===============================
def compute_distance(row):
    return geodesic(user_location, (row["latitude"], row["longitude"])).km

df_day["distance"] = df_day.apply(compute_distance, axis=1)

# ===============================
# NORMALISATION
# ===============================
features = ["distance", "traffic", "patients_per_month", "days_since_last_visit"]
scaler = MinMaxScaler()
df_day[[f+"_norm" for f in features]] = scaler.fit_transform(df_day[features])

# ===============================
# SCORE
# ===============================
df_day["distance_score"] = 1 - df_day["distance_norm"]
df_day["traffic_score"] = 1 - df_day["traffic_norm"]

df_day["score"] = (
    0.25 * df_day["distance_score"] +
    0.25 * df_day["traffic_score"] +
    0.30 * df_day["patients_per_month_norm"] +
    0.20 * df_day["days_since_last_visit_norm"]
)

# ===============================
# CLUSTERING 🤖
# ===============================
cluster_features = df_day[["patients_per_month", "traffic", "distance"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df_day["cluster"] = kmeans.fit_predict(cluster_features)

labels = {0:"🔥 High",1:"⚖️ Medium",2:"❄️ Low"}
df_day["cluster_label"] = df_day["cluster"].map(labels)

# ===============================
# TOP AUTO
# ===============================
top = df_day.sort_values("score", ascending=False).head(n_pharma)

# ===============================
# COMBINE AUTO + MANUEL
# ===============================
if len(selected_pharmacies) > 0:
    manual = df_region[df_region["name"].isin(selected_pharmacies)]
    combined = pd.concat([top, manual]).drop_duplicates(subset="name")
else:
    combined = top.copy()

if combined.empty:
    st.warning("Aucune pharmacie sélectionnée")
    st.stop()

# ===============================
# 🚀 FAST ROUTE (GREEDY)
# ===============================
remaining = combined.copy()
current = user_location
route = []

while len(remaining) > 0:
    remaining["dist"] = remaining.apply(
        lambda row: geodesic(current, (row["latitude"], row["longitude"])).km,
        axis=1
    )

    next_point = remaining.loc[remaining["dist"].idxmin()]
    route.append(next_point)

    current = (next_point["latitude"], next_point["longitude"])
    remaining = remaining.drop(next_point.name)

route_df = pd.DataFrame(route)

# ===============================
# MAP 📍
# ===============================
m = folium.Map(location=user_location, zoom_start=12)

# position actuelle
folium.Marker(user_location, popup="Position actuelle", icon=folium.Icon(color="green")).add_to(m)

coords = [user_location]

for i, row in route_df.iterrows():
    loc = (row["latitude"], row["longitude"])
    coords.append(loc)

    folium.Marker(loc, popup=f"{i+1}. {row['name']}", icon=folium.Icon(color="blue")).add_to(m)

folium.PolyLine(coords, color="red").add_to(m)

map_data = st_folium(m, width=800, height=500)

# ===============================
# CLICK TEMP 📍
# ===============================
if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    st.session_state["temp_location"] = (clicked["lat"], clicked["lng"])

# afficher temporaire
if st.session_state["temp_location"] is not None:
    st.info(f"📍 Position sélectionnée (non validée): {st.session_state['temp_location']}")

    if st.button("✅ Valider cette position"):
        st.session_state["user_location"] = st.session_state["temp_location"]
        st.session_state["temp_location"] = None
        st.success("📍 Position mise à jour !")
        st.rerun()

# ===============================
# 📊 GRAPH INDIVIDUEL
# ===============================
st.subheader("📊 Analyse trafic")

selected_graph = st.selectbox("Choisir pharmacie", route_df["name"])

pharma_data = route_df[route_df["name"] == selected_graph].iloc[0]

hours = [f"hour_{i}" for i in range(24)]
values = pharma_data[hours].values

fig, ax = plt.subplots()
ax.plot(range(24), values, marker='o')

ax.set_title(f"Trafic - {selected_graph}")
ax.set_xlabel("Heure")
ax.set_ylabel("Traffic")

st.pyplot(fig)

# ===============================
# DISPLAY
# ===============================
st.subheader("🏆 Pharmacies sélectionnées")
st.dataframe(combined[["name", "score", "cluster_label"]])

st.subheader("🧭 Route optimisée")
st.dataframe(route_df[["name"]])