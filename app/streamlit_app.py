import os.path
import folium
import math
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

# CSS to inject that hides row index when displaying Pandas dataframe in Streamlit app
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_state_locations():
    """
    Load the statelatlong.csv file as a GeoPandas dataframe

    Returns:
        Contents of statelatlong.csv as a GeoPandas dataframe
    """
    file_path = os.path.join(os.getcwd(), "app", "data", "statelatlong.csv")
    state_coordinates = pd.read_csv(file_path)
    df = gpd.GeoDataFrame(state_coordinates, geometry=gpd.points_from_xy(state_coordinates.Longitude, state_coordinates.Latitude))
    df.dropna(inplace=True)
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_locations():
    """
    Load the us_hospital_locations.csv file as Pandas dataframe

    Returns:
        Contents of us_hospital_locations.csv as a Pandas dataframe
    """
    file_path = os.path.join(os.getcwd(), "app", "data", "us_hospital_locations.csv")
    hospital_coordinates = pd.read_csv(file_path)
    df = gpd.GeoDataFrame(hospital_coordinates, geometry=gpd.points_from_xy(hospital_coordinates.LONGITUDE, hospital_coordinates.LATITUDE))
    df.dropna(inplace=True)
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_ratings():
    """
    Load CMS hospital ratings data setusing the CMS API, filtering the data to only keep hospitals with available
    ratings and emergency services

    Returns:
        Data retrieved from the CMS API for hospital ratings as a Pandas dataframe
    """
    df = query_cms_api("https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0/download?format=csv")
    df = df.iloc[:, :13].drop(columns=['phone_number', 'meets_criteria_for_promoting_interoperability_of_ehrs'])
    df = df[df["hospital_overall_rating"] != "Not Available"]
    df["hospital_overall_rating"] = df["hospital_overall_rating"].astype(int)
    df = df[df['emergency_services'] == 'Yes']
    return df


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def load_hospital_survey():
    """
    Load CMS hospital survey data set using the CMS API. Calculates positive response rate for given set of questions
    binned by type (doctors, nurses, patients, staff)

    Returns:
        Data retrieve and processed from CMS API for hospital survey as a Pandas dataframe
    """
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
    """
    Merge hospital rating and survey dataframes using facility_id as the key to a single dataframe

    Parameters:
        ratings: hospital ratings Pandas dataframe
        survey: hospital survey Pandas dataframe

    Returns:
        Pandas dataframe
    """
    merged = ratings.merge(survey, on="facility_id").dropna().reset_index()
    merged = merged.drop(labels=['index', 'hospital_type', 'hospital_ownership'], axis=1)
    return merged


@st.cache(ttl=3*60*60, suppress_st_warning=True, allow_output_mutation=True)
def merge_hospital_location_ratings(locations, ratings):
    """
    Merge hospital rating and locations dataframes using facility name and state as key to a single dataframe

    Parameters:
        locations: hospital locations Pandas dataframe
        ratings: hospital ratings Pandas dataframe

    Returns:
        Pandas dataframe
    """
    merged = locations.merge(ratings, left_on=["NAME", "STATE"], right_on=["facility_name", "state"])
    return merged


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def gather_covid_data():
    """
    Gather U.S county-level COVID-19 data using the Arcgis web API to pull down the data

    Returns:
        COVID-19 data in a GeoPandas dataframe
    """
    url = 'https://services5.arcgis.com/qWZ7BaZXaP5isnfT/arcgis/rest/services/Community_Profile_Report_Counties/FeatureServer/0/'
    gdf = query_arcgis_feature_server(url)
    gdf = gdf[~gdf.County.str.startswith("Unallocated")]
    return gdf


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def random_query_generator(hospitals, n=100):
    """
    Random query generator to generate queries to be used to test recommendation system

    Parameters:
        hospitals: hospital information as a Pandas dataframe
        n: number of random queries to be generated

    Returns:
        Pandas dataframe of random queries that will be used to test the recommendation system
    """
    qs = [[random.choice(hospitals["state"].unique()),
           random.randint(int(hospitals["doctors"].min()), 100),
           random.randint(int(hospitals["nurses"].min()), 100),
           random.randint(int(hospitals["staffs"].min()), 100),
           random.randint(int(hospitals["patients"].min()), 100)] for i in range(n)]
    df_queries = pd.DataFrame(qs,
                              columns=['selected_state',
                                       'doctor_rating',
                                       'nurses_rating',
                                       'staff_rating',
                                       'patient_rating'])
    return df_queries


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def generate_recs_base_for_rand_queries(query_list, hospitals):
    """
    Generate recommendations for a list of queries, which are actually a Pandas dataframe

    Parameters:
        query_list: random queries as a Pandas dataframe
        hospitals: hospital information as a Pandas dataframe

    Returns:
        Dictionary of recommendations and top rated hospitals for each query
    """
    query_rec_dict = {}
    for i in range(len(query_list)):
        query = query_list.iloc[i].to_dict()
        recommendations = recommend_hospitals(hospitals, query)
        hosp_rel = hospitals[hospitals['state'] == query['selected_state']]
        hosp_rel = hosp_rel.sort_values(by=['hospital_overall_rating'], ascending=False)
        hosp_rel = hosp_rel[hosp_rel['hospital_overall_rating'] == hosp_rel['hospital_overall_rating'].values.max()]
        query_rec_dict["query_{}".format(i)] = [recommendations, hosp_rel]
    return query_rec_dict


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def evaluation_pre_rec(queries, survey_ratings, n=-1):
    """
    Calculate precision and recall at n for each query in queries

    Parameters:
        queries: Pandas dataframe of random queries that will be used to test the recommendation system
        survey_ratings: Pandas dataframe of hospital survey ratings and other information
        n: Top n results to use to calculate precision and recall per query

    Returns:
        Two lists, one containing the calculated precision of each query and one for the calculated recall of each query
    """
    pre_at_n, rec_at_n = [], []
    query_rec_base = generate_recs_base_for_rand_queries(queries, survey_ratings)
    for q_id, recs in query_rec_base.items():
        recommendations = recs[0]
        hosp_rel = recs[1]
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


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def evaluation_mean_avg_pre(queries, survey_ratings, cutoff=-1):
    """
    Calculate (mean) average precision for each query in queries

    Parameters:
        queries: Pandas dataframe of random queries that will be used to test the recommendation system
        survey_ratings: Pandas dataframe of hospital survey ratings and other information
        cutoff: Top n results to use to calculate average precision per query

    Returns:
        List of average precisions for each query and the mean average precision
    """
    avg_pre = []
    all_pres = []
    query_rec_base = generate_recs_base_for_rand_queries(queries, survey_ratings)
    for q_id, recs in query_rec_base.items():
        precisions = []
        recommendations = recs[0]
        hosp_rel = recs[1]
        retrieved = recommendations
        not_retrieved = []
        if (cutoff != -1) and (cutoff <= len(recommendations)):
            retrieved = recommendations[:cutoff]
            not_retrieved = recommendations[cutoff:]
        for j in range(1, len(retrieved) + 1):
            docs = retrieved[:j]
            if docs.iloc[-1]['facility_id'] in hosp_rel['facility_id'].tolist():
                numerator = len(set(docs['facility_id']).intersection(hosp_rel['facility_id']))
                if numerator != 0:
                    precisions.append(numerator/len(docs))
                else:
                    precisions.append(0)
        for doc in not_retrieved:
            if doc in hosp_rel:
                precisions.append(0)
        avg_pre.append(sum(precisions) / len(hosp_rel))
        all_pres.append(sum(precisions) / len(hosp_rel))
    mean_avg_pre = sum(all_pres) / len(all_pres)
    return avg_pre, mean_avg_pre


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def evaluation_ndcg(queries, survey_ratings, n=-1, base=2):
    """
    Calculate NDCG at n for each query in queries

    Parameters:
        queries: Pandas dataframe of random queries that will be used to test the recommendation system
        survey_ratings: Pandas dataframe of hospital survey ratings and other information
        n: Top n results to use to calculate nDCG per query
        base: Base of the logarithm function used to discount relevance scores

    Returns:
        List of nDCGs for each query
    """
    ndcg = []
    query_rec_base = generate_recs_base_for_rand_queries(queries, survey_ratings)
    for q_id, recs in query_rec_base.items():
        recommendations = recs[0]
        hosp_rel = recs[1]
        j_doc_scores = list(range(len(hosp_rel) + 1, 1, -1))
        if len(recommendations) > len(j_doc_scores):
            j_doc_scores = j_doc_scores + ([1]*(len(recommendations) - len(j_doc_scores)))
        retrieved_scores = []
        for doc in recommendations["facility_id"].tolist():
            counter = 0
            if doc in hosp_rel["facility_id"].tolist():
                retrieved_scores.append(j_doc_scores[hosp_rel["facility_id"].tolist().index(doc)])
            else:
                retrieved_scores.append(1)
            counter += 1
        if (n != -1) and (n <= len(recommendations)):
            retrieved_scores = retrieved_scores[:n]
        j_doc_scores = j_doc_scores[:len(retrieved_scores)]
        rs_sum = 0
        for j, val in enumerate(retrieved_scores):
            if j+1 < base:
                rs_sum += val
            else:
                rs_sum += (val / math.log(j+1, base))
        jd_sum = 0
        for k, val in enumerate(j_doc_scores):
            if k+1 < base:
                jd_sum += val
            else:
                jd_sum += (val / math.log(k+1, base))
        ndcg.append(rs_sum/jd_sum)
    return ndcg


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def add_metrics_to_queries(queries, pre_at_n, rec_at_n, avg_pre, ndcg):
    """
    Add the evaluation metrics to the queries dataframe

    Parameters:
        queries: Pandas dataframe of random queries that was used to test the recommendation system
        pre_at_n: List of precisions (one entry per query)
        rec_at_n: List or recalls (one entry per query)
        avg_pre: List of average precisions (one entry per query)
        ndcg: List of nDCG (one entry per query)

    Returns:
        Pandas dataframe containing queries and metrics
    """
    query_metrics = queries.copy(deep=True)
    query_metrics["Precision"] = pre_at_n
    query_metrics["Recall"] = rec_at_n
    query_metrics["Average Precision"] = avg_pre
    query_metrics["nDCG"] = ndcg
    return query_metrics


def recommend_hospitals(hospitals, user, num_recommendations=5):
    """
    Generate hospital recommendations using cosine similarity.
    Hospitals are first filtered by the user-specified state and then cosine similarity is taken between user entered
    parameters and hospital ratings for doctor_rating, nurses_rating, staff_rating, and patient_rating.

    Parameters:
        hospitals: Pandas dataframe of hospital survey ratings and other information
        user: Dictionary of user data containing parameters specified by the user (state, doctor_rating, nurses_rating, staff_rating, patient_rating)
        num_recommendations: Number of recommendations to generate

    Returns:
        Pandas dataframe of the top num_recommendations recommended hospitals
    """
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


def main():
    st.set_page_config(
        page_title="Hospital Recommendation Engine to Receive Optimal Care"
    )
    st.title("Hospital Recommendation Engine to Receive Optimal Care")
    st.header("Anurag Bolneni, Brian Minie, Ridima Bhatt")

    st.header("I. Introduction")
    st.markdown(
        """
        U.S. healthcare is one of the most convoluted sectors; transparency and ease of access remains scant. For our 
        project, we assessed CMS data to help recommend hospitals for a patient, based on ratings they desire for different 
        parameters (e.g. doctor, nurses). Furthermore, we???ve included COVID-19 prevalence in different regions to add 
        additional layers of choice for a patient; they can actively avoid hospitals in areas of COVID-19 if that???s a major 
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
        new recommendations are requested we can quickly make and present new recommendations in 1-2 seconds instead of 
        having to reload and process all the required data which would take approximately 1-2 minute.
        ### Static Data Sets
        * statelatlong.csv
            * United States latitude and longitude coordinates that will be used as default state locations when plotting recommended hospital locations when no COVID-19 data is to be plotted. Originally downloaded from https://www.kaggle.com/datasets/washimahmed/usa-latlong-for-state-abbreviations
        * us_hospital_locations.csv
            * Information about 7596 hospitals in the United States including latitude, longitude, staff, beds, ownership, etc. All records were originally extracted from the U.S. Department of Homeland Security with the complete dataset downloaded from https://www.kaggle.com/datasets/andrewmvd/us-hospital-locations  
        ### Dynamic Data Sets
        * CMS datasets
            * Hospital General Information
                * List of all Medicare listed hospitals with geographic and overall hospital metrics
                * Accessed via CMS's provided API: https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0/download?format=csv
            * Patient Survey (HCAHPS)
                * List of all hospital ratings per patient survey regarding their recent inpatient hospital stay
                * Accessed via CMS's provided API: https://data.cms.gov/provider-data/api/1/datastore/query/dgck-syfz/0/download?format=csv
        * COVID-19 dataset
            * U.S county-level COVID-19 data for the past 7 days. 
            * Accessed via the HHS Public Protect Hub API: https://protect-public.hhs.gov/datasets/cad5b70395a04b95936f0286be30e282/api
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
    st.caption("Example of hospital general information dataframe")
    st.markdown(
        """
        #### Patient Survey (HCAHPS)
        The CMS HCAHPS patient survey data is a national, standardized survey of hospital patients attaining ratings about 
        their recent inpatient hospital stay experiences. Each hospital facility ID has 22 distinct measures tracking 
        patient experience. The patient experiences questions can be broken down into twelve different categories of 
        ratings per Facility ID and obtain metrics on patients. The data dictionary describing the survey fields can be 
        found on Page 90 of https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/hospital/HospitalCompare-DataDictionary.pdf
        """)
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
    st.caption("Measures used to assess doctors, nurses, patients, and staff ratings for each hospital")
    st.markdown(
        """
        For each hospital, we took the mean of sub parameters to determine an overall score for each parameter. These four 
        measures then became hospital parameters for our recommendation engine to score against patient inputs.
        """)
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(hospital_survey.head())
    st.caption("Example of hospital survey dataframe")
    st.markdown(
        """
        ### Combining Data Sets
        After cleaning the hospital general information and patient survey data, we joined the two datasets on Facility ID. 
        This enabled us to centralize all of our model parameters and necessary hospital information within a single 
        dataframe. 
        When we began cleaning the data we realized that duplicate addresses exist but each Facility ID is unique; in 
        total, this amounted to 5306 unique hospital Facility IDs. Upon further evaluation of the hospital type, we 
        recognized that many of the duplicate addresses have psychiatry or non-invasive specialties that would not be 
        appropriate for evaluation anyway given we???re focused on in-patient hospital care. Given this information, we 
        combined the datasets and dropped rows with NA values.
        
        We initially also considered using data on all US hospitals from the American Hospital Directory (AHD), but upon 
        cleaning realized there were too many discrepancies between AHD and CMS.
        """)

    st.header("III. Hospital Recommendation Engine")
    st.markdown(
        """
        Our hospital recommendation engine first begins by asking the user to enter their recommendation preferences. We 
        ask for user to specify their state and their ideal ratings for doctors, nurses, staff, and patients to discern the 
        types of hospitals a user would like. We collect these preferences as our query vector and use them to generate 
        hospital recommendations.
        
        Our hospital recommendation engine has a two-step process: filter the hospital data based on the user's selected 
        state and calculate cosine similarity using the user's ideal ratings for doctors, nurses, staff, and patients and 
        the same measurements forall filtered hospitals to return those hospitals that are most similar (i.e. highest 
        cosine similarity) with the user's preferences. The hospital recommendations (i.e. those with the 5 highest cosine 
        similarity) are collected and presented to the user along with a map of their locations as well as county-level 
        COVID-19 data for the user selected state if the user requests it. An interactive example is provided in section V. 
        Mapping Recommended Hospital Locations and Ease of Practical Use.
        """)

    st.header("IV. Recommendation System Evaluation and Metrics")
    st.markdown(
        """
        Our recommendation system is a retrieval system based on ranking of hospitals calculated using their cosine 
        similarity with respect to the user specified query. In order to test its effectiveness, we explored a few 
        evaluation metrics that would be suitable for such a system, namely Precision, Recall, Mean Average Precision (MAP) 
        and Normalized Discounted Cumulative Gain (nDCG).
        
        We have created a random query generator to simulate queries and used a set of 5000 such artificially generated 
        queries for this evaluation. The relevance base that the recommendations are compared against is the CMS top rated 
        hospitals that offer emergency services.
        """)
    queries = random_query_generator(survey_ratings, 5000)
    pre_at_n, rec_at_n = evaluation_pre_rec(queries, survey_ratings, 10)
    avg_pre, mean_avg_precision = evaluation_mean_avg_pre(queries, survey_ratings, 10)
    ndcg = evaluation_ndcg(queries, survey_ratings)
    queries_metrics = add_metrics_to_queries(queries, pre_at_n, rec_at_n, avg_pre, ndcg)
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(queries_metrics.head(5))
    st.caption("Example of collected performance metrics for test queries")
    st.markdown(
        """
        Mean Average Precision: {}
        """.format(mean_avg_precision))
    pre_hist = alt.Chart(queries_metrics).mark_bar().encode(
        alt.X("Precision:Q", bin=True),
        y="count()",
    ).properties(title="Histogram of Precision for 5000 Test Queries")
    rec_hist = alt.Chart(queries_metrics).mark_bar().encode(
        alt.X("Recall:Q", bin=True),
        y="count()",
    ).properties(title="Histogram of Recall for 5000 Test Queries")
    avg_pre_hist = alt.Chart(queries_metrics).mark_bar().encode(
        alt.X("Average Precision:Q", bin=True),
        y="count()",
    ).properties(title="Histogram of Average Precision for 5000 Test Queries")
    scatter = alt.Chart(queries_metrics).mark_line().encode(
        alt.X("Recall:Q"),
        alt.Y("Precision:Q"),
    ).properties(title="Precision-Recall Curve for 5000 Test Queries")
    ndcg_hist = alt.Chart(queries_metrics).mark_bar().encode(
        alt.X("nDCG:Q", bin=True),
        y="count()",
    ).properties(title="Histogram nDCG for 5000 Test Queries")
    st.altair_chart((pre_hist | rec_hist) & (scatter | avg_pre_hist) & ndcg_hist, use_container_width=True)
    st.caption("Compiled performance metrics for test queries")
    st.markdown(
        """        
        Precision and Recall for majority of our test queries was less than 0.1 while the metrics for the other half of 
        the batch were spread unevenly across the remainder of the range. Along similar lines, the Average Precision 
        for ~60% of the test queries was 0.1 or lower, resulting in a Mean Average Precision of only 0.16 for our 
        system. The nDCG results, however, were spread more evenly across the range in comparison even though the 
        majority score was 0.5 or less, indicating that our system provided only about 50% of the best ranking possible 
        on an average.
        
        These results indicate that our hospital recommendations are not particularly in line with the hospitals ranked 
        highest as per the CMS ratings. This could be due to additional survey parameters that we have taken into 
        account as they might differ in comparison to the hospital overall rating being used as the base.
        """)

    st.header("V. Mapping Recommended Hospital Locations and Ease of Practical Use")
    st.markdown(
        """
        The streamlit app itself actually pulls double duty, it functions as a blog post as well as an interactive 
        application for users to get hospital recommendations. Once the app has been started, there are various sections 
        that detail the background of the project, evaluation metrics, etc. One of those sections is the section where 
        users can enter their inputs and receive recommendations. The user will select their state from a dropdown menu and 
        then entire their ideal rating for doctors, nurses, staff, and what other patients have rated the hospital on a 
        scale of 1-100. Then users will hit the "Generate Recommendations" button and their recommendations will be made and 
        displayed to the user. This includes information about the recommended hospitals as well as a map of the hospital 
        locations. There is an additional option that the user can use to specify if they want COVID-19 data overlaid on 
        the map. Depending on the selection, the map may contain the county-level COVID-19 data for the past 7 days along 
        with an example of the information that is available for each county. The map is built using Folium and uses the 
        streamlit-folium extension to display is properly to the user. The map is interactive as well so users can zoom 
        in/out as well as use the tooltips to see additional information about the hospitals and COVID-19 data per county. 
        Users can change inputs and regenerated new recommendations at any time just by changing their inputs in the form 
        and hitting "Generate Recommendations" again.
        
        To test this functionality, you can use the form provided below. The page will be updated any time you hit the 
        Generate Recommendations button.
        """)
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
        st.markdown(
            """
            In this section you will find an interactive map showing the recommended hospital locations COVID-19 data will 
            be deployed depending on your answer to "Do you want to see COVID-19 data by county?"
            """)
        m = folium.Map()
        if display_covid == "Yes":
            m = community_covid.explore(column="Cases_last_7_days", legend=True, tooltip=["County",
                                                                                          "Cases_last_7_days",
                                                                                          "Deaths_last_7_days",
                                                                                          "Cases_percent_change",
                                                                                          "Cumulative_cases",
                                                                                          "Cumulative_deaths"])
        else:
            m = folium.Map(location=[state_location["Latitude"], state_location["Longitude"]], zoom_start=6)
        if len(recommended) != 0:
            recommended.apply(lambda row: folium.Marker(location=[row["LATITUDE"], row["LONGITUDE"]],
                                                        tooltip="<b>{}</b><br><b>{}</b><br><b>{},{}</b><br>".format(row["NAME"],
                                                                                                                    row["ADDRESS"],
                                                                                                                    row["CITY"],
                                                                                                                    row["STATE"])).add_to(m), axis=1)
        folium_static(m)
        st.caption("Map of recommended hospitals with/without COVID-19 data overlay")

        st.subheader("Information on Recommended Hospitals")
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(recommended[["NAME", "ADDRESS", "CITY", "STATE", "TELEPHONE", "WEBSITE", "hospital_overall_rating",
                              "Cosine Similarity"]])
        st.caption("Information of recommended hospitals")

        if display_covid == "Yes":
            st.subheader("COVID-19 Information By County")
            st.markdown(
                """
                In this section you will find information on COVID-19 if you selected "Yes" to the question "Do you want to 
                see COVID-19 data by county?" We are only displaying the first 5 counties in the state so you can get a 
                sense as to what the data looks like.
                """)
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(community_covid.drop("geometry", axis=1)[:5])
            st.caption("Example of COVID-19 data by county in selected state")

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
        Currently, our model accounts for 14/22 questions asked within the CMS HCAHPS survey. We could increase additional 
        model parameters to capture all survey responses and further isolate buckets of questions. For instance, there are 
        a number of patient ratings about receiving timely medication, which can become a standalone parameter. Other 
        parameters could include overall hospital, comprising patient ratings on hospital cleanliness, ambiance, and 
        quietness.
        
        Our current model does not take into account user location which could result in the recommended hospital being a 
        long distance from the users location (ex. user lives in northern California but based on their specified 
        parameters the top recommended hospital is in southern California). By capturing the users location we could add the 
        commute distance to the different hospitals as a new model parameter to improve our recommendations.
        """)
    st.subheader("Enhancing Evaluation Metrics")
    st.markdown(
        """
        The current evaluation metrics are calculated against the CMS overall hospital rating. Additionally, the model 
        could be evaluated against top hospitals by distance. Zip codes are also available in the general information 
        dataset so they could be used as an additional parameter to incorporate distance into our system and enhance the 
        quality of recommendations generated. The HCAHPS sub-parameters that we have incorporated into the model could also 
        be normalized to obtain more accurate and realistic results.
        """)
    st.subheader("Adapting Model to Users and Events")
    st.markdown(
        """
        Our hospital recommendation engine can be modified for a range of events and users. For instance, the COVID-19 
        overlay can be changed to another global pandemic???s data should it become prevalent. Furthermore, we can wrap our 
        model in an LSTM by capturing user feedback on the accuracy of our model and refining it in real-time accordingly. 
        This would likely require setting up AWS servers to retain data and run a cloud cluster for LSTM.
        """)

    st.header("VII: Statement of Work")
    st.markdown(
        """
        Anurag originally came up with the idea for this project and acted as the subject matter expert. He also identified 
        the CMS data sets to use as well as generating the recommender system MVP which included data cleaning, merging, 
        and exploration. Anurag also provided feedback on the streamlit app design and acted as the overall project lead.
        
        Brian focused on streamlining data processing, cleaning, and merging as well as implementing usage of public API's 
        to pull down data and converting the recommender system MVP into the final recommender system deployed as part of 
        this post/app. Brian also designed the Streamlit app, handled visualizations and hospital location mappings, 
        integrated retrieval and plotting of COVID-19 data, and managed the GitHub repo and conda environment used as part 
        of this project. 
        
        Brian and Ridima implemented the recommender system evaluation metrics. Ridima also worked on data cleaning and 
        exploration as well as designing the random query generator used as part of the recommender system evaluation. 
        Ridima also provided feedback on the streamlit app design.
        
        All team members contributed equally to writing the final report/blog post.
        """)


if __name__ == '__main__':
    main()
