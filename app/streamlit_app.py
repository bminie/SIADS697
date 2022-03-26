import os.path
import folium
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from gatherData import query_arcgis_feature_server
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Hospital Recommendation Engine",
    layout="wide"
)
st.title("Hospital Recommendation Engine")


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_state_locations():
    file_path = os.path.join(os.getcwd(), "data", "statelatlong.csv")
    state_coordinates = pd.read_csv(file_path)
    df = gpd.GeoDataFrame(state_coordinates, geometry=gpd.points_from_xy(state_coordinates.Longitude, state_coordinates.Latitude))
    df.dropna(inplace=True)
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_locations():
    file_path = os.path.join(os.getcwd(), "data", "us_hospital_locations.csv")
    hospital_coordinates = pd.read_csv(file_path)
    df = gpd.GeoDataFrame(hospital_coordinates, geometry=gpd.points_from_xy(hospital_coordinates.LONGITUDE, hospital_coordinates.LATITUDE))
    df.dropna(inplace=True)
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_ratings():
    file_path = os.path.join(os.getcwd(), "data", "Hospital_General_Information.csv")
    df = pd.read_csv(file_path).iloc[:, :13].drop(columns=['Phone Number', 'Meets criteria for promoting interoperability of EHRs'])
    df = df[df["Hospital overall rating"] != "Not Available"]
    df["Hospital overall rating"] = df["Hospital overall rating"].astype(int)
    #df['Emergency Services'] = [1 if x == 'Yes' else 0 for x in df['Emergency Services']]
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True, allow_output_mutation=True)
def merge_hospital_locaion_ratings(locations, ratings):
    merged = locations.merge(ratings, left_on="NAME", right_on="Facility Name")
    return merged


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def gather_covid_data():
    url = 'https://services5.arcgis.com/qWZ7BaZXaP5isnfT/arcgis/rest/services/Community_Profile_Report_Counties/FeatureServer/0/'
    gdf = query_arcgis_feature_server(url)
    gdf = gdf[~gdf.County.str.startswith("Unallocated")]
    return gdf


def generate_static_map(community_covid, selected_hospitals):
    # Older function to generate a pyplot static map, may want to delete this later if we do not use it
    f, ax = plt.subplots(1, figsize=(20, 12))
    ax = community_covid.plot(column="Cases_last_7_days", legend=True, missing_kwds={'color': 'lightgrey'}, ax=ax)
    selected_hospitals.plot(ax=ax, color="red")
    f.suptitle("COVID-19 Cases in the Last 7 Days in {} By County".format(selected_state))
    return f


#def recommend_hospitals(hospitals, patient):


state_locations = load_state_locations()
hospital_gdf = load_hospital_locations()
hospital_ratings = load_hospital_ratings()
merged = merge_hospital_locaion_ratings(hospital_gdf, hospital_ratings)

st.sidebar.text("Please Select Your Recommendation \nParameters")
selected_state = st.sidebar.selectbox(
    "Select the state of interest",
    sorted(state_locations.State.unique())
)
rating = st.sidebar.slider("Select minimum hospital rating", 0, 5)
emergency_services = st.sidebar.selectbox(
    "Do you need emergency services?",
    ["Yes", "No"]
)
display_covid = st.sidebar.selectbox(
    "Do you want to see COVID-19 data by county?",
    ["Yes", "No"]
)

community_data = gather_covid_data()
community_covid = community_data[community_data.State_Abbreviation == selected_state]
state_location = state_locations[state_locations["State"] == selected_state]
selected_hospitals = merged[merged['STATE'] == selected_state]
selected_hospitals = selected_hospitals[selected_hospitals["Hospital overall rating"] >= rating]
selected_hospitals = selected_hospitals[selected_hospitals["Emergency Services"] == emergency_services]

st.subheader("Information on Recommended Hospitals")
st.text("In this section you will find information on the hospitals that are recommended to you")
st.table(selected_hospitals[["NAME","ADDRESS","CITY","STATE","TELEPHONE"]])

st.subheader("Map of Recommended Hospitals")
st.text('In this section you will find an interactive map showing the recommended hospital locations\n'
        'COVID-19 data will be deployed depending on your answer to "Do you want to see COVID-19 data by county?"')
m = folium.Map()
if display_covid == "Yes":
    m = community_covid.explore(column="Cases_last_7_days", legend=True)
else:
    m = folium.Map(location=[state_location["Latitude"], state_location["Longitude"]], zoom_start=6)
if len(selected_hospitals) != 0:
    selected_hospitals.apply(lambda row:folium.Marker(location=[row["LATITUDE"], row["LONGITUDE"]],
                                                      tooltip=row["NAME"]).add_to(m), axis=1)
folium_static(m)

if display_covid == "Yes":
    st.subheader("COVID-19 Information By County")
    st.text('In this section you will find information on COVID-19 if you selected "Yes" to the question "Do you want '
            'to see COVID-19 data by county?"')
    st.table(community_covid.drop("geometry", axis=1))
