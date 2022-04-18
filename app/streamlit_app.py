import os.path
import folium
import random
import altair as alt
import pandas as pd
import streamlit as st
import geopandas as gpd
from streamlit_folium import folium_static
from sklearn.metrics.pairwise import cosine_similarity
from gatherData import *
pd.set_option('mode.chained_assignment', None)
random.seed(42)

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """


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
def merge_hospital_location_ratings(locations, ratings):
    merged = locations.merge(ratings, left_on=["NAME", "STATE"], right_on=["facility_name", "state"])
    return merged


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def gather_covid_data():
    url = 'https://services5.arcgis.com/qWZ7BaZXaP5isnfT/arcgis/rest/services/Community_Profile_Report_Counties/FeatureServer/0/'
    gdf = query_arcgis_feature_server(url)
    gdf = gdf[~gdf.County.str.startswith("Unallocated")]
    return gdf


def recommend_hospitals(hospitals, user, num_recommendations=5):
    collab_filtered = hospitals[hospitals['state'] == user["selected_state"]]
    hospital_arr = collab_filtered[['doctors', 'nurses', 'staffs', 'patients']].to_numpy()
    user_arr = np.array([user['doctor_rating'],
                         user['nurses_rating'],
                         user['staff_rating'],
                         user['patient_rating']]).reshape(1, -1)
    cosim = cosine_similarity(user_arr, hospital_arr)[0]
    collab_filtered['Cosine Similarity'] = cosim
    final = collab_filtered.sort_values('Cosine Similarity', ascending=False).reset_index().iloc[0:num_recommendations]
    return final


def random_query_generator(state_abbreviations, n=100):
    qs = [[random.choice(state_abbreviations),
           random.randint(70, 100),
           random.randint(50, 100),
           random.randint(0, 100),
           random.randint(0, 100)] for i in range(n)]
    df_queries = pd.DataFrame(qs,
                              columns=['selected_state', 'doctor_rating', 'nurses_rating', 'staff_rating', 'patient_rating'])
    return df_queries

@st.cache(ttl=3*60*60, suppress_st_warning=True)
def evaluation_pre_rec(queries, hospitals, n=-1):
    pre_at_n, rec_at_n = [], []
    for i in range(len(queries)):
        query = queries.iloc[i].to_dict()
        recommendations = recommend_hospitals(hospitals, query)
        hosp_rel = hospitals[hospitals['state'] == query['selected_state']]
        hosp_rel = hosp_rel.sort_values(by=['hospital_overall_rating'], ascending=False)
        hosp_rel = hosp_rel[hosp_rel['hospital_overall_rating'] == hosp_rel['hospital_overall_rating'].values.max()]
        if (n != -1) and (n <= len(recommendations)):
            recommendations = recommendations.iloc[:n]
        numerator = len(set(recommendations['facility_id']).intersection(hosp_rel['facility_id']))
        if numerator != 0:
            pre_at_n.append(numerator / len(recommendations))
            rec_at_n.append(numerator / len(hosp_rel))
        else:
            pre_at_n.append(0)
            rec_at_n.append(0)
    return pre_at_n, rec_at_n


st.set_page_config(
    page_title="Hospital Recommendation Engine to Receive Optimal Care"
)
st.title("Hospital Recommendation Engine to Receive Optimal Care")
st.header("Anurag Bolneni, Brian Minie, Ridima Bhatt")

st.header("I. Introduction")
st.markdown(
    """
    U.S. healthcare is of the most convoluted sectors; transparency and ease of access remains scant. For our project, 
    we assessed CMS data to help recommend hospitals for a patient, based on ratings they desire for different 
    parameters (e.g. doctor, nurses). Furthermore, we’ve included COVID-19 prevalence in different regions to add 
    additional layers of choice for a patient; they can actively avoid hospitals in areas of COVID-19 if that’s a major 
    concern. 
    
    Our analysis has five key components:
    * Data Exploration
    * Hospital Recommendation Engine
    * Evaluation Metrics
    * Visualization and Ease of Practical Use
    * Future Directions
    
    We hope our hospital recommendation engine helps increase visibility into the U.S. hospital system for patients, 
    helping patients make more informed decisions about their healthcare.
    """
)

st.header("II. Data Exploration")
st.markdown(
    """
    There are a variety of data sources that are used as part of this application, some that are static and do not 
    change (these can be found in the app/data directory) and others that are gathered each time the application is 
    started as this data is updated regularly. These dynamic data sets are accessed via API's provide by the different 
    data providers and loaded into the application when it first starts. The purpose of this methodology is to ensure 
    the most recent available data is always used and to limit the number of file read/writes that are required to 
    access the data required by the application.
    
    There were some challenges that arose with using the API's that we had to work around, namely rate limits for the 
    API's  we are using. To work around the CMS API's rate limit, instead of accessing via API and getting a JSON 
    response which was limited to 1000 records and would have required multiple repeated API calls, we instead use the 
    API that returns a CSV file as the response. By using the API in this way, we get back all the data we need with a 
    single request and then easily convert the response to a Pandas dataframe for additional processing. The Public 
    Protect Hub ArcGIS REST API for accessing the COVID-19 data had to be treated in a slightly different manner. 
    This API had a hard limit on a maximum of 1,000 records return per API call. Since each item in the response is the 
    county-level COVID-19 data, with over 1,000 counties in the United States we had to find a way around this record 
    limit. Our solution was to create a function that downloads all the data on the feature server and handles a couple 
    of common pitfalls such as avoiding the maximum record limit and request that are too big/time out. Using this 
    function, we are able to request the entire county-level COVID-19 data set and load it to a GeoPandas dataframe 
    that we use for plotting the data via Folium. 
    
    In this app, all data is cached once it has been loaded. By caching the data instead of access the files any time 
    new recommendations are requests we can quickly make new recommendations in 1-2 seconds instead of having to reload 
    and process all the required data which would take approximately 1-2 minute.
    ### Static Data Sets
    * statelatlong.csv
        * United States latitude and longitude coordinates that will be used as default state locations when plotting recommended hospital locations when no COVID-19 data is to be plotted. Originally downloaded from https://www.kaggle.com/datasets/washimahmed/usa-latlong-for-state-abbreviations
    * us_hospital_locations.csv
        * Information about 7596 hospitals in the United States including latitude, longitude, staff, beds, ownership, etc. All records were originally extracted from the U.S. Department of Homeland Security with the complete dataset downloaded from https://www.kaggle.com/datasets/andrewmvd/us-hospital-locations  
    ### Dynamic Data Sets
    * CMS datasets
        * Hospital General Information
            * List of all Medicare listed hospitals with geographic and overall hospital metrics
        * Patient Survey (HCAHPS)
            * List of all hospital ratings per patient survey regarding their recent inpatient hospital stay
    * COVID-19 dataset
    """)
state_locations = load_state_locations()
hospital_gdf = load_hospital_locations()
hospital_ratings = load_hospital_ratings()
hospital_survey = load_hospital_survey()
location_ratings = merge_hospital_location_ratings(hospital_gdf, hospital_ratings)
survey_ratings = merge_hospital_rating_survey(location_ratings, hospital_survey)
st.markdown(
    """
    #### Hospital General Information
    The CMS Hospital General Information has a number of fields from CMS including addresses, phone numbers, hospital 
    type, and overall hospital rating. We processed this dataframe to eliminate extraneous fields and kept geographic, 
    contact, and overall hospital information (e.g. rating, emergency services). This helped us retain a dataframe 
    with only the necessary fields of information.
    """)
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(hospital_ratings.head())
st.markdown(
    """
    #### Patient Survey (HCAHPS)
    The CMS HCAHPS patient survey data is a national, standardized survey of hospital patients attaining ratings about 
    their recent inpatient hospital stay experiences. Each hospital facility ID has 24 distinct measures tracking 
    patient experience. The patient experiences questions can be broken down into twelve different categories of 
    ratings per Facility ID and obtain metrics on patients. Examples of these measures can be seen below:
    """)
# PLACEHOLDER FOR DATA DICTIONARY IMAGE
#st.image()
st.markdown(
    """
    We then looked through each of the Measures, which contain sub-questions. For our hospital recommendation model, we 
    were interested in Composite 1, 2, 3, and 5. In total, this amounts to four patient-centric model parameters, which 
    consist of 14 sub-parameters. The full list of parameters we used can be found below.
    """)
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(pd.DataFrame(data={"Nurses": ["H_COMP_1_A_P", "H_NURSE_RESPECT_A_P", "H_NURSE_LISTEN_A_P", "H_NURSE_EXPLAIN_A_P"],
                            "Doctors": ["H_COMP_2_A_P", "H_DOCTOR_RESPECT_A_P", "H_DOCTOR_LISTEN_A_P", "H_DCOTOR_EXPLAIN_A_P"],
                            "Patients": ["H_COMP_3_A_P", "H_CALL_BUTTON_A_P", "H_BATH_HELP_A_P", ""],
                            "Staffs": ["H_COMP_5_A_P", "H_MED_FOR_A_P", "H_SIDE_EFFECTS_A_P", ""]}))
st.markdown(
    """
    For each hospital, we took the mean of sub parameters to determine an overall score for each parameter. These four 
    measures then became hospital parameters for our recommendation engine to score against patient inputs.
    """)
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(hospital_survey.head())
st.markdown(
    """
    ### Combining Data Sets
    After cleaning the hospital general information and patient survey data, we joined the two datasets on Facility ID. 
    This enabled us to centralize all of our model parameters and necessary hospital information within a single 
    dataframe. 
    When we began cleaning the data we realized that duplicate addresses exist but each Facility ID is unique; in 
    total, this amounted to 5306 unique hospital Facility IDs. Upon further evaluation of the hospital type, we 
    recognized that many of the duplicate addresses have psychiatry or non-invasive specialties that would not be 
    appropriate for evaluation anyway given we’re focused on in-patient hospital care. Given this information, we 
    combined the datasets and dropped rows with NA values.
    
    We initially also considered using data on all US hospitals from the American Hospital Directory (AHD), but upon 
    cleaning realized there were too many discrepancies between AHD and CMS.
    """)

st.header("III. Hospital Recommendation Engine")
st.markdown(
    """
    Our hospital recommendation engine first begins by asking the user to enter their recommendation preferences. We 
    ask for user to specify their state and their ideal ratings for doctors, nurses, staff, and patients to discern the 
    types of hospitals a user would like. We collect these preferences to a dictionary and use to generate 
    recommendations of optimal hospitals.
    
    Our hospital recommendation engine has a two-step process: filter the hospital data based on the user's selected 
    state and conduct cosine similarity for the user's ideal ratings for doctors, nurses, staff, and patients against 
    all filtered hospitals to return those hospitals that are most similar (i.e. highest cosine similarity) with the 
    user's preferences. The hospital recommendations are collected and presented to the user along with a map of their 
    locations. An interactive example is provided in section V. Mapping Recommended Hospital Locations and Ease of 
    Practical Use.
    """)

st.header("IV. Recommendation System Evaluation and Metrics")
# Add in support for displaying recommendation system results
queries = random_query_generator(state_locations["State"], 5000)
pre_at_n, rec_at_n = evaluation_pre_rec(queries, survey_ratings, 10)
queries["Precision"] = pre_at_n
queries["Recall"] = rec_at_n
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(queries.head(5))
pre_hist = alt.Chart(queries).mark_bar().encode(
    alt.X("Precision:Q", bin=True),
    y="count()",
).properties(title="Histogram of Precision for 5000 Test Queries")
rec_hist = alt.Chart(queries).mark_bar().encode(
    alt.X("Recall:Q", bin=True),
    y="count()",
).properties(title="Histogram of Recall for 5000 Test Queries")
st.altair_chart((pre_hist | rec_hist))

st.header("V. Mapping Recommended Hospital Locations and Ease of Practical Use")
community_data = gather_covid_data()
st.subheader("Please Select Your Recommendation Parameters")
with st.form(key="my_form"):
    selected_state = st.selectbox(
        "Select the state of interest",
        sorted(state_locations.State.unique())
    )
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
    recommended = recommend_hospitals(survey_ratings,
                                      {"selected_state": selected_state,
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
    if len(recommended) != 0:
        recommended.apply(lambda row: folium.Marker(location=[row["LATITUDE"], row["LONGITUDE"]],
                                                    tooltip=row["NAME"]).add_to(m), axis=1)
    folium_static(m)

    st.subheader("Information on Recommended Hospitals")
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(recommended[["NAME", "ADDRESS", "CITY", "STATE", "TELEPHONE", "WEBSITE", "hospital_overall_rating",
                          "Cosine Similarity"]])

    if display_covid == "Yes":
        st.subheader("COVID-19 Information By County")
        st.markdown(
            """
            In this section you will find information on COVID-19 if you selected "Yes" to the question "Do you want to see COVID-19 data by county?"
            """)
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(community_covid.drop("geometry", axis=1))
st.header("VI. Future Directions")
st.markdown(
    """
    The nice part about our model is that it can be readily adapted, dynamically. Future directions could include 
    increasing model parameters, enhancing evaluation metrics, and even adapting our model to user feedback and 
    real-world events.
    """)
st.subheader("Increasing Model Parameters")
st.markdown(
    """
    Currently, our model accounts for 14/24 questions asked within the CMS HCAHPS survey. We could increase additional 
    model parameters to capture all survey responses and further isolate buckets of questions. For instance, there are 
    a number of patient ratings about receiving timely medication, which can become a standalone parameter. Other 
    parameters could include overall hospital, comprising patient ratings on hospital cleanliness, ambiance, and 
    quietness.
    """)
st.subheader("Enhancing Evaluation Metrics")
st.markdown("""""")
st.subheader("Adapting Model to Users and Events")
st.markdown(
    """
    Our hospital recommendation engine can be modified for a range of events and users. For instance, the COVID-19 
    overlay can be changed to another global pandemic’s data should it become prevalent. Furthermore, we can wrap our 
    model in an LSTM by capturing user feedback on the accuracy of our model and refining it in real-time accordingly. 
    This would likely require setting up AWS servers to retain data and run a cloud cluster for the LSTM, though.
    """)
