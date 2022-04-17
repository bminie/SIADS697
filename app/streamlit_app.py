import os.path
import folium
import random
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from gatherData import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Hospital Recommendation Engine",
    layout="wide"
)
st.title("Hospital Recommendation Engine")
expander = st.expander("Learn More")
expander.write(
    """
    Elective procedures in the US have been frequently canceled over the past 2 years since COVID began; just as things were returning to normal, hospitals are now facing capacity issues and canceling elective procedures. 
    However, disease has no regard for COVID and continues to adversely human health, untreated. 
    Receiving healthcare is already an overly opaque and complex process - recent events have only made things more difficult. 
    We've created an open-source, web-based tool to help people navigate which US hospitals to get treatment at for elective procedures based on value-based care measures and convenience. 
    Our recommendation engine takes user input (e.g. type of procedure needed, location, etc), along with weighted user priorities for model features (e.g. cost of care, quality of care, etc), to provide a “top 3 hospitals” to attain care with regard to COVID-19 (or future disease) prevalence in various geographies.
    
    To use this application, enter the requested information on the left. When the app first starts up, all required data will be retrieved, loaded, and cached (i.e. saved) to save on resources and compute time.
    Each time you change one of the parameters, your recommendations and the results displayed will be update.
    The question "Do you want to see COVID-19 data by county?" has no effect on the recommendatons, just what is displayed on the map.
    """
)


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
    df = query_cms_api("https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0/download?format=csv")
    df = df.iloc[:, :13].drop(columns=['phone_number', 'meets_criteria_for_promoting_interoperability_of_ehrs'])
    df = df[df["hospital_overall_rating"] != "Not Available"]
    df["hospital_overall_rating"] = df["hospital_overall_rating"].astype(int)
    df = df[df['emergency_services'] == 'Yes']
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_survey():
    question_type_dict = {'H_COMP_1_A_P': "nurses",
                          'H_NURSE_RESPECT_A_P': "nurses",
                          'H_NURSE_LISTEN_A_P': "nurses",
                          'H_NURSE_EXPLAIN_A_P': "nurses",
                          'H_COMP_2_A_P': "doctors",
                          'H_DOCTOR_RESPECT_A_P': "doctors",
                          'H_DOCTOR_LISTEN_A_P': "doctors",
                          'H_DOCTOR_EXPLAIN_A_P': "doctors",
                          'H_COMP_3_A_P': "patients",
                          'H_CALL_BUTTON_A_P': "patients",
                          'H_BATH_HELP_A_P': "patients",
                          'H_COMP_5_A_P': "staffs",
                          'H_MED_FOR_A_P': "staffs",
                          'H_SIDE_EFFECTS_A_P': "staffs"
                          }
    df = query_cms_api("https://data.cms.gov/provider-data/api/1/datastore/query/dgck-syfz/0/download?format=csv")
    df = df[['facility_id', 'hcahps_measure_id', 'hcahps_question', 'hcahps_answer_percent']]
    df['hcahps_answer_percent'] = pd.to_numeric(df['hcahps_answer_percent'], errors='coerce')
    df = df.dropna(axis=0)
    counts = df.groupby(['facility_id']).count()
    filtered_counts = counts[counts['hcahps_question'] == 72].reset_index()
    valid_facility_id = list(filtered_counts['facility_id'])
    df = df[df['facility_id'].isin(valid_facility_id)]
    df["measurement_type"] = df.apply(lambda row: question_type_dict[row["hcahps_measure_id"]] if row["hcahps_measure_id"] in question_type_dict.keys() else "UNKNOWN", axis=1)
    grouped = df.groupby(['facility_id', 'measurement_type']).mean()
    grouped = grouped.drop("UNKNOWN", level="measurement_type").reset_index()
    pivot = grouped.pivot(index="facility_id", columns="measurement_type", values="hcahps_answer_percent").reset_index()
    pivot.columns.name = None
    return pivot


@st.cache(ttl=3*60*60, suppress_st_warning=True, allow_output_mutation=True)
def merge_hospital_rating_survey(ratings, survey):
    merged = ratings.merge(survey, on="facility_id").dropna().reset_index()
    merged = merged.drop(labels=['index', 'hospital_type', 'hospital_ownership'], axis=1)
    return merged


@st.cache(ttl=3*60*60, suppress_st_warning=True, allow_output_mutation=True)
def merge_hospital_locaion_ratings(locations, ratings):
    merged = locations.merge(ratings, left_on="NAME", right_on="facility_name")
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


def recommend_hospitals(hospitals, user):
    collab_filtered = hospitals[hospitals['state'] == user["selected_state"]]
    #collab_filtered = collab_filtered[collab_filtered["hospital_overall_rating"] >= user["hospital_rating"]]
    #collab_filtered = collab_filtered[collab_filtered["emergency_services"] == user["emergency_services"]]
    hospital_arr = collab_filtered[['doctors', 'nurses', 'staffs', 'patients']].to_numpy()
    user_arr = np.array([user['doctor_rating'],
                         user['nurses_rating'],
                         user['staff_rating'],
                         user['patient_rating']]).reshape(1, -1)
    cosim = cosine_similarity(user_arr, hospital_arr)[0]
    collab_filtered['Cosine Similarity'] = cosim
    return collab_filtered.sort_values('Cosine Similarity', ascending=False).reset_index().iloc[0:10]


def random_query_generator(state_abbreviations, n=5):
    qs = [[random.choice(state_abbreviations),
                random.randint(70, 100),
                random.randint(50, 100),
                random.randint(0, 100),
                random.randint(0, 100)] for i in range(n)]
    df_queries = pd.DataFrame(qs,
                              columns=['selected_state', 'doctor_rating', 'nurses_rating', 'staff_rating', 'patient_rating'])
    return df_queries


def evaluation_pre_rec(queries, hospitals, n=5):
    pre_at_n, rec_at_n = {}, {}
    for i in range(len(queries)):
        query = queries.iloc[i].to_dict()
        recommendations = recommend_hospitals(hospitals, query)
        print('query #' + str(i + 1) + ': ' + str(len(recommendations)) + ' recommendations generated')

        hosp_rel = hospitals[(hospitals['state'] == query['selected_state'])]
        hosp_rel = hosp_rel.replace('Not Available', np.nan)
        hosp_rel = hosp_rel.sort_values(by=['hospital_overall_rating'], ascending=False)
        hosp_rel = hosp_rel[hosp_rel['hospital_overall_rating'] == hosp_rel['hospital_overall_rating'].values.max()]
        if (n != -1) and (n <= len(recommendations)):
            recommendations = recommendations[:n]
        numerator = len(set(recommendations['facility_id']).intersection(hosp_rel['facility_id']))
        if numerator != 0:
            pre_at_n['q' + str(i + 1)] = numerator / len(recommendations)
            rec_at_n['q' + str(i + 1)] = numerator / len(hosp_rel)
        else:
            pre_at_n['q' + str(i + 1)] = 0
            rec_at_n['q' + str(i + 1)] = 0
    return pre_at_n, rec_at_n


state_locations = load_state_locations()
hospital_gdf = load_hospital_locations()
hospital_ratings = load_hospital_ratings()
hospital_survey = load_hospital_survey()
location_ratings = merge_hospital_locaion_ratings(hospital_gdf, hospital_ratings)
survey_ratings = merge_hospital_rating_survey(hospital_ratings, hospital_survey)
community_data = gather_covid_data()

st.sidebar.text("Please Select Your Recommendation \nParameters")
with st.sidebar.form(key="my_form"):
    selected_state = st.selectbox(
        "Select the state of interest",
        sorted(state_locations.State.unique())
    )
    #emergency_services = st.selectbox(
    #    "Do you need emergency services?",
    #    ["Yes", "No"]
    #)
    #rating = st.slider("Select minimum hospital rating", 1, 5)
    doctor_rating = st.slider("Specify your ideal doctor rating", 1, 100)
    nurses_rating = st.slider("Specify your ideal nurses rating", 1, 100)
    staff_rating = st.slider("Specify your ideal staff rating", 1, 100)
    patient_rating = st.slider("Specify your ideal patient rating", 1, 100)
    display_covid = st.selectbox(
        "Do you want to see COVID-19 data by county?",
        ["Yes", "No"]
    )
    pressed = st.form_submit_button("Generate Recommendations")

community_covid = community_data[community_data.State_Abbreviation == selected_state]
state_location = state_locations[state_locations["State"] == selected_state]

if pressed:
    selected_hospitals = recommend_hospitals(survey_ratings, {#"hospital_rating": rating,
                                         #"emergency_services": emergency_services,
                                         "selected_state": selected_state,
                                         "doctor_rating": doctor_rating,
                                         "nurses_rating": nurses_rating,
                                         "patient_rating": patient_rating,
                                         "staff_rating": staff_rating})

    st.subheader("Map of Recommended Hospitals")
    st.text('In this section you will find an interactive map showing the recommended hospital locations\n'
            'COVID-19 data will be deployed depending on your answer to "Do you want to see COVID-19 data by county?"')
    m = folium.Map()
    if display_covid == "Yes":
        m = community_covid.explore(column="Cases_last_7_days", legend=True)
    else:
        m = folium.Map(location=[state_location["Latitude"], state_location["Longitude"]], zoom_start=6)
    if len(selected_hospitals) != 0:
        selected_locations = selected_hospitals.merge(location_ratings, on="facility_id")
        selected_locations.apply(lambda row: folium.Marker(location=[row["LATITUDE"], row["LONGITUDE"]],
                                                           tooltip=row["NAME"]).add_to(m), axis=1)
    folium_static(m)

    st.subheader("Information on Recommended Hospitals")
    st.text("In this section you will find information on the hospitals that are recommended to you")
    st.table(selected_hospitals)

    if display_covid == "Yes":
        st.subheader("COVID-19 Information By County")
        st.text(
            'In this section you will find information on COVID-19 if you selected "Yes" to the question "Do you want '
            'to see COVID-19 data by county?"')
        st.table(community_covid.drop("geometry", axis=1))

    # Add in support for displaying recommendation system results
    queries = random_query_generator(state_locations["State"], 100)
    pre_at_n, rec_at_n = evaluation_pre_rec(queries, survey_ratings, 10)
    st.dataframe(queries)
    st.write("Precision: {}".format(pre_at_n))
    st.write("Recall: {}".format(rec_at_n))
