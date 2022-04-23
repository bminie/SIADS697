# Team Care: Hospital Recommendation Engine to Receive Optimal Care
## Overview
U.S. healthcare is one of the most convoluted sectors; transparency and ease of access remains scant. For our 
project, we assessed CMS data to help recommend hospitals for a patient, based on ratings they desire for different 
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
## Getting Started
### Accessing the Pre-Built Streamlit App
This application has been built with Streamlit and is hosted on Streamlit Cloud. If you'd like to access the app, click 
the badge below and you will be taken to the app. This is provided as an alternative to running the app locally and 
makes it available to the wider community and those with the expertise to get the app running on their local machine.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/bminie/siads697/main/app/streamlit_app.py)
### Create Virtual Environment
It's highly recommended that you use a virtual environment when working with the app with conda being recommended.

Using either the requirements.txt or environment.yml file, create a virtual environment using conda
```commandline
conda env create -f conda_environment.yml python=3.9
conda create --name team_care --file requirements.txt python=3.9
```
### Starting the Streamlit App
You can use the following commands to start the streamlit app.

The first command is used to activate the conda environment. If you followed the instructions above then your 
environment will be named team_care. If you gave your environment an alternative name, update the conda activate 
command to use your conda environment.

The command may different slightly depending on your OS and conda version. Update the command appropriately for your OS 
and conda version.
```commandline
conda activate team_care
streamlit run app/streamlit_app.py
```
## Application Data Overview
There are a variety of data sources that are used as part of this application, some that are static and do not change 
(these can be found in the app/data directory) and others that are gathered each time the application is started as 
this data is updated regularly. These dynamic data sets are accessed via API's provide by the different data providers 
and loaded into the application when it first starts. The purpose of this methodology is to ensure the most recent 
available data is always used and to limit the number of file read/writes that are required to access the data required 
by the application.

There were some challenges that arose with using the API's that we had to work around, namely rate limits for the API's 
we are using. To work around the CMS API's rate limit, instead of accessing via API and getting a JSON response which 
was limited to 1000 records and would have required multiple repeated API calls, we instead use the API that returns a 
CSV file as the response. By using the API in this way, we get back all the data we need with a single request and then 
easily convert the response to a Pandas dataframe for additional processing. The Public Protect Hub ArcGIS REST API for 
accessing the COVID-19 data had to be treated in a slightly different manner. This API had a hard limit on a maximum of 
1,000 records return per API call. Since each item in the response is the county-level COVID-19 data, with over 1,000 
counties in the United States we had to find a way around this record limit. Our solution was to create a function that 
downloads all the data on the feature server and handles a couple of common pitfalls such as avoiding the maximum record
limit and request that are too big/time out. Using this function, we are able to request the entire county-level 
COVID-19 data set and load it to a GeoPandas dataframe that we use for plotting the data via Folium. This specific 
function in the source code is query_arcgis_feature_server in app/gatherData.py

In this app, all data is cached once it has been loaded. By caching the data instead of access the files any time new 
recommendations are requests we can quickly make new recommendations in 1-2 seconds instead of having to reload and 
process all the required data which would take approximately 1-2 minute.
### Static Data Sets
* statelatlong.csv
  * United States latitude and longitude coordinates that will be used as default state locations when plotting 
recommended hospital locations when no COVID-19 data is to be plotted. 
  * Originally downloaded from 
https://www.kaggle.com/datasets/washimahmed/usa-latlong-for-state-abbreviations
* us_hospital_locations.csv
  * Information about 7596 hospitals in the United States including latitude, longitude, staff, beds, ownership, etc.
  * All records were originally extracted from the U.S. Department of Homeland Security with the complete dataset downloaded
from https://www.kaggle.com/datasets/andrewmvd/us-hospital-locations
### Dynamic Data Sets
* Hospital General Information
  * List of all Medicare listed hospitals with geographic and overall hospital metrics
  * Accessed via CMS's provided API: https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0/download?format=csv
* Patient Survey (HCAHPS)
  * List of all hospital ratings per patient survey regarding their recent inpatient hospital stay
  * Accessed via CMS's provided API: https://data.cms.gov/provider-data/api/1/datastore/query/dgck-syfz/0/download?format=csv
* COVID-19 dataset
  * U.S county-level COVID-19 data for the past 7 days. 
  * Accessed via the HHS Public Protect Hub API: https://protect-public.hhs.gov/datasets/cad5b70395a04b95936f0286be30e282/api
#### Hospital General Information
The CMS Hospital General Information has a number of fields from CMS including addresses, phone numbers, hospital type, 
and overall hospital rating. We processed this dataframe to eliminate extraneous fields and kept geographic, contact, 
and overall hospital information (e.g. rating, emergency services). This helped us retain a dataframe with only the 
necessary fields of information.
#### Patient Survey (HCAHPS)
The CMS HCAHPS patient survey data is a national, standardized survey of hospital patients attaining ratings about 
their recent inpatient hospital stay experiences. Each hospital facility ID has 24 distinct measures tracking patient 
experience. The patient experiences questions can be broken down into twelve different categories of ratings per 
Facility ID and obtain metrics on patients. The data dictionary describing the survey fields can be found at 
https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/hospital/HospitalCompare-DataDictionary.pdf

We then looked through each of the Measures, which contain sub-questions. For our hospital recommendation model, we 
were interested in Composite 1, 2, 3, and 5. In total, this amounts to four patient-centric model parameters, which 
consist of 14 sub-parameters. The full list of parameters we used can be found below.

For each hospital, we took the mean of sub parameters to determine an overall score for each parameter. These four 
measures then became hospital parameters for our recommendation engine to score against patient inputs.
### Combining Data Sets
After cleaning the hospital general information and patient survey data, we joined the two datasets on Facility ID. 
This enabled us to centralize all of our model parameters and necessary hospital information within a single dataframe. 

When we began cleaning the data we realized that duplicate addresses exist but each Facility ID is unique; in total, 
this amounted to 5306 unique hospital Facility IDs. Upon further evaluation of the hospital type, we 
recognized that many of the duplicate addresses have psychiatry or non-invasive specialties that would not be 
appropriate for evaluation anyway given we’re focused on in-patient hospital care. Given this information, we 
combined the datasets and dropped rows with NA values.
    
We initially also considered using data on all US hospitals from the American Hospital Directory (AHD), but upon 
cleaning realized there were too many discrepancies between AHD and CMS.
## Hospital Recommendation Engine
Our hospital recommendation engine first begins by asking the user to enter their recommendation preferences. We 
ask for user to specify their state and their ideal ratings for doctors, nurses, staff, and patients to discern the 
types of hospitals a user would like. We collect these preferences as our query vector and use them to generate 
hospital recommendations.
    
Our hospital recommendation engine has a two-step process: filter the hospital data based on the user's selected 
state and calculate cosine similarity using the user's ideal ratings for doctors, nurses, staff, and patients and 
the same measurements for all filtered hospitals to return those hospitals that are most similar (i.e. highest 
cosine similarity) with the user's preferences. The hospital recommendations (i.e. those with the 5 highest cosine 
similarity) are collected and presented to the user along with a map of their locations as well as county-level 
COVID-19 data for the user selected state if the user requests it.
## Recommendation System Evaluation and Metrics
Our recommendation system is a retrieval system based on ranking of hospitals calculated using their cosine 
similarity with respect to the user specified query. In order to test its effectiveness, we explored a few 
evaluation metrics that would be suitable for such a system, namely Precision, Recall, Mean Average Precision (MAP) 
and Normalized Discounted Cumulative Gain (nDCG).
    
We have created a random query generator to simulate queries and used a set of 5000 such artificially generated 
queries for this evaluation. The relevance base that the recommendations are compared against is the CMS top rated 
hospitals that offer emergency services.

Precision and Recall for more than half our test queries was less than 0.1 while the metrics for the other half of 
the batch were spread unevenly across the remainder of the range. Along similar lines, the MAP for ~70% of the test 
queries was 0.1 or lower. However, the nDCG results were spread more evenly across the range in comparison even 
though the majority score was less than 0.5, indicating that our system provided only about 50% of the best ranking 
possible.
    
These results indicate that our hospital recommendations are not particularly in line with the hospitals ranked 
highest as per the CMS ratings. This could be due to additional survey parameters that we have taken into account 
as they might differ in comparison to the hospital overall rating being used as the base.
## Mapping Recommended Hospital Locations and Ease of Practical Use
The streamlit app itself actually pulls double duty, it functions as a blog post as well as an interactive application 
for users to get hospital recommendations. Once the app has been started, there are various sections that detail the 
background of the project, evaluation metrics, etc. One of those sections is the section where users can enter their 
inputs and receive recommendations. The user will select their state from a dropdown menu and then entire their ideal 
rating for doctors, nurses, staff, and what other patients have rated the hospital on a scale of 1-100. Then users will 
hit the "Generate Recommendations" button and their recommendations will be made and displayed to the user. This includes 
information about the recommended hospitals as well as a map of the hospital locations. There is an additional option 
that the user can use to specify if they want COVID-19 data overlaid on the map. Depending on the selection, the map 
may contain the county-level COVID-19 data for the past 7 days along with an example of the information that is 
available for each county. The map is built using Folium and uses the streamlit-folium extension to display is properly 
to the user. The map is interactive as well so users can zoom in/out as well as use the tooltips to see additional 
information about the hospitals and COVID-19 data per county. Users can change inputs and regenerated new 
recommendations at any time just by changing their inputs in the form and hitting "Generate Recommendations" again.

https://user-images.githubusercontent.com/38255038/164472751-e6a8bbb5-1e13-406c-b2a7-4d46a217ac8e.mov
## Future Directions
The nice part about our model is that it can be readily adapted, dynamically. Future directions could include 
increasing model parameters, enhancing evaluation metrics, and even adapting our model to user feedback and real-world 
events.
### Application Performance Enhancements
Currently, in the app we are loading and calculating performance metrics for 5000 test queries to assess our information 
retrieval and recommendation system. While this is done only upon app start-up it can take a few minutes to complete. 
The main reason for this is that the performance metrics are calculated in a serial fashion based on the same 5000 test 
queries. One way to improve this is to calculate these performance metrics in parallel which will significantly cut 
down on the initial app start-up time. This can be done in a variety of ways, from using Python's built-in parallel 
processing features likes Process and Queue to creating a pipeline that can be run as part of the app and generates the 
performance metrics. 
### Increasing Model Parameters
Currently, our model accounts for 14/24 questions asked within the CMS HCAHPS survey. We could increase additional 
model parameters to capture all survey responses and further isolate buckets of questions. For instance, there are 
a number of patient ratings about receiving timely medication, which can become a standalone parameter. Other 
parameters could include overall hospital, comprising patient ratings on hospital cleanliness, ambiance, and 
quietness.
    
Our current model does not take into account user location which could result in the recommended hospital being a 
long distance from the user's location (ex. user lives in northern California but based on their specified 
parameters the top recommended hospital is in southern California). By capturing the users location we could add the 
commute distance to the different hospitals as a new model parameter to improve our recommendations.
### Enhancing Evaluation Metrics
The current evaluation metrics are calculated against the CMS overall hospital rating. Additionally, the model 
could be evaluated against top hospitals by distance. Zip codes are also available in the general information 
dataset so they could be used as an additional parameter to incorporate distance into our system and enhance the 
quality of recommendations generated. The HCAHPS sub-parameters that we have incorporated into the model could also 
be normalized to obtain more accurate and realistic results.
### Adapting Model to Users and Events
Our hospital recommendation engine can be modified for a range of events and users. For instance, the COVID-19 overlay 
can be changed to another global pandemic’s data should it become prevalent. Furthermore, we can wrap our model in an 
LSTM by capturing user feedback on the accuracy of our model and refining it in real-time accordingly. This would 
likely require setting up AWS servers to retain data and run a cloud cluster for LSTM.
