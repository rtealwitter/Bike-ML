# Bike-ML
Final project for Professor Mohri's Fall 2020
Foundations of Machine Learning.

# Files
* `weather.csv` is the daily weather data from the station at JFK from 1/1/2017 to 12/31/2019
and includes columns `AWND` through `WSF2`
* `combined1.csv` (formerly `combined_cb_ct_collision_0.csv`) contains columns `datetime` through `ncoll`
* `combined2.csv` is the [inner_join](https://www.rdocumentation.org/packages/plyr/versions/1.8.6/topics/join)
of `combined1.csv` and weather on `standarddate`

# Transferring Large Files
GitHub only allows files of size 50MB so we had to 
split and compress `combined1.csv` and `combined2.csv`.
In order to reconstruct both files, you must run
`bash handledata.sh` from the command line in the
top level directory.


# Columns
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
* `age_bins_44-65_out`
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
