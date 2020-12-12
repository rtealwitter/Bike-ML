# Bike-ML
Final project for Professor Mohri's Fall 2020
Foundations of Machine Learning.

# Goal
The objective is to predict when and where
bike collisions occur in Manhattan.
Each observation in our dataset corresponds to a zone
(several blocks) at a particular hour in either 2018 or 2019.
We would like to classify each observation as either
containing a collision in that zone and hour or not.
As you might expect, the classes are highly imbalanced.
Of the nearly 3,000,000 observations, only about
1,800 contain collisions.
A machine learning model could take advantage of the
discrepancy and simply by predicting no collision
could achieve an error rate less than .1%.
Therefore we rely on a series of data preparation
techniques (e.g. resampling, reweighting)
and models (e.g. logistic regression, linear SVM,
neural network, custom aggregation model, and boosting)
to achieve a lower error rate on the observations
with collisions while simultaneously supressing the
error rate on observations without collisions.

# Data
* `weather.csv` is the daily weather data from the station at JFK from 1/1/2017 to 12/31/2019
and includes columns `AWND` through `WSF2`
* `data.csv` contains the full data set (including `weather.csv`)
* `smalltrain.csv` contains 100,000 randomly selected observations used for training while working on the implementation
* `smalltest.csv` contains 100,000 randomly selected observations used for testing while working on the implementation

# Building the data files.
GitHub only allows files of size 50MB so we had to 
split and compress `data.csv` into three files.
In order to reconstruct `data.csv`, you must run
`bash handledata.sh` from the command line in the
top level of the repository.

# Running the Code
Run `python3 experiment.py` from the command line
in the top level of the repository.
As currently written, the program will run on the full dataset
which is *very* computationally intensive.
Switching the flat `usefulldata` to `False` while
run the training and testing on the `smalltrain.csv` and
`smalltest.csv` instead.

# Features
* `datetime`
* `GEOID`: tract ID, combination of state, county, and area code (**turn this into binary features**)
* `date`
* `month`
* `hour`
* `intrips`: Number of trips ending in the zone
* `usertype_Customer_in`
* `usertype_Subscriber_in`
* `age_bins_U18_in`
* `age_bins_18-24_in`
* `age_bins_25-44_in`
* `age_bins_44-65_in`
* `age_bins_44-65_in`
* `outtrips`: Number of trips starting in the zone
* `usertype_Customer_out`
* `usertype_Subscriber_out`
* `age_bins_U18_out`
* `age_bins_18-24_out`
* `age_bins_25-44_out`
* `age_bins_44-65_out`
* `age_bins_65P_out`
* `day`
* `year`
* `speed_mph_mean`: Mean of Uber's speeds in the GEOID zone (not always available)
* `speed_mph_stddev`: Standard deviation of Uber's speeds in the GEOID zone (not always available)
* `standarddate`
* `NAME_1`: Name of GEOID
* `ncoll`: Number of collisions (**variable we will predict**)
* `AWND`: Average daily wind speed (miles per hour)
* `PRCP`: Precipitation (inches)
* `SNOW`: Snowfall (inches)
* `SNWD`: Snow depth (inches)
* `TAVG`: Average hourly temperature (Fahrenheit)
* `TMAX`: Maximum hourly temperature (Fahrenheit)
* `TMIN`: Minimum hourly temperature (Fahrenheit)
* `WDF2`: Direction of fastest 2-minute wind (degrees)
* `WSF2`: Fastest 2-minute wind speed (miles per hour)
* `standarddate`: Standardized date format
