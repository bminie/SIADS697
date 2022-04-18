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
### Streamlit App Usage
The streamlit app itself actually pulls double duty, it functions as a blog post as well as an interactive application 
for users to get hospital recommendations. Once the app has been started, there are various sections that detail the 
background of the project, evaluation metrics, etc. One of those sections is the section where users can enter their 
inputs and receive recommendations.
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
* CMS datasets
  * CMS Stars
  * Patient Survey (HCAHPS)
* COVID-19 dataset

Current data in data folder was acquired on February 23, 2022 by manually downloading files. In the future, these files should be automatically pulled down using API's or other access methods in order to be working off the most recent data.
* Community_Profile_Report_Counties.csv
    * County level COVID-19 data pulled from https://protect-public.hhs.gov/
* Community_Profile_Report_Counties.zip
    * County level COVID-19 data pulled from https://protect-public.hhs.gov/ . This is a zipped GEOJSON file that includes coordinates to draw counties on a map. File as zipped due to file size restrictions with standard GitHub account
* hospitalizations.csv
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* hospitalizations.json
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* hospitalizations.xlsx
    * Hospitalization data collected from state websites by the COVID-19 Hospitalization Tracking Project Team, pulled from https://carlsonschool.umn.edu/mili-misrc-covid19-tracking-project/download-data
* Data from The Centers for Medicare & Medicaid Services website
    * Hospital Readmissions Reduction Program: https://data.cms.gov/provider-data/dataset/9n3s-kdb3
    * Hospital-Acquired Condition (HAC) Reduction Program: https://data.cms.gov/provider-data/dataset/yq43-i98g
    * Unplanned Hospital Visits - Hospital: https://data.cms.gov/provider-data/dataset/632h-zaca
    * Payment and value of care - Hospital: https://data.cms.gov/provider-data/dataset/c7us-v4mf
    * Patient survey (HCAHPS) - Hospital: https://data.cms.gov/provider-data/dataset/dgck-syfz
    * Hospital Value-Based Purchasing (HVBP) - Efficiency Scores:https://data.cms.gov/provider-data/dataset/su9h-3pvj
    * CMS Stars: https://data.cms.gov/provider-data/dataset/xubh-q36u
## Recommendation System Overview
Our recommendations are based off of the cosine similarity measurement. Using the user inputs as our query vector, we 
calculate the cosine similarity for all hospitals in the state selected by the user and return our top 5 as our 
recommendation.
## Future Directions