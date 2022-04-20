# Team Care: Hospital Recommendation Engine to Receive Optimal Care
## Overview
Elective procedures in the US have been frequently canceled over the past 2 years since COVID began; just as things were returning to normal, hospitals are now facing capacity issues and canceling elective procedures. 
However, disease has no regard for COVID and continues to adversely human health, untreated. 
Receiving healthcare is already an overly opaque and complex process - recent events have only made things more difficult. 
We've created an open-source, web-based tool to help people navigate which US hospitals to get treatment at for elective procedures based on value-based care measures and convenience. 
Our recommendation engine takes user input (e.g. type of procedure needed, location, etc), along with weighted user priorities for model features (e.g. cost of care, quality of care, etc), to provide a “top 3 hospitals” to attain care with regard to COVID-19 (or future disease) prevalence in various geographies.

To build the web-based tool we are using the Streamlit python library as it enables rapid prototyping while producing production-quality dashboards. User will easily be able to enter parameters into the web app and visualize the results via tables and maps.
## Getting Started
### Create Virtual Environment
It's highly recommended that you use a virtual environment when working with the app with conda being recommended.

Using either the requirements.txt or environemnt.yml file, create a virtual environment using conda
```commandline
conda env create -f environment.yml
conda create --name <env_name> --file requirements.txt
```
### Starting the Streamlit App
Ensure that you are in the app directory and run the following command to start the streamlit app.

The first command to activate the virtual environment is what will be used if you used the environment.yml to create the conda environment.

The command may different slightly depending on your OS and conda version. Update the command appropriately for your OS and conda version.
```commandline
conda activate team_care
streamlit run streamlit_app.py
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
recommended hospital locations when no COVID-19 data is to be plotted. Originally downloaded from 
https://www.kaggle.com/datasets/washimahmed/usa-latlong-for-state-abbreviations
* us_hospital_locations.csv
  * Information about 7596 hospitals in the United States including latitude, longitude, staff, beds, ownership, etc. 
All records were originally extracted from the U.S. Department of Homeland Security with the complete dataset downloaded
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
## Hospital Recommendation Engine
Our hospital recommendation engine first begins by asking the user to enter their recommendation preferences. We 
ask for user to specify their state and their ideal ratings for doctors, nurses, staff, and patients to discern the 
types of hospitals a user would like. We collect these preferences as our query vector and use them to generate 
hospital recommendations.
    
Our hospital recommendation engine has a two-step process: filter the hospital data based on the user's selected 
state and calculate cosine similarity using the user's ideal ratings for doctors, nurses, staff, and patients and 
the same measurements forall filtered hospitals to return those hospitals that are most similar (i.e. highest 
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
inputs and receive recommendations.
## Future Directions
The nice part about our model is that it can be readily adapted, dynamically. Future directions could include 
increasing model parameters, enhancing evaluation metrics, and even adapting our model to user feedback and real-world 
events.
### Increasing Model Parameters
Currently, our model accounts for 14/24 questions asked within the CMS HCAHPS survey. We could increase additional 
model parameters to capture all survey responses and further isolate buckets of questions. For instance, there are 
a number of patient ratings about receiving timely medication, which can become a standalone parameter. Other 
parameters could include overall hospital, comprising patient ratings on hospital cleanliness, ambiance, and 
quietness.
    
Our current model does not take into account user location which could result in the recommended hospital being a 
long distance from the users location (ex. user lives in northern California but based on their specified 
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