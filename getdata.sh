#!/bin/bash

download () {
    if curl --head -silent --fail $3; then
    if test -f "trips/$2.csv"; then
    echo "$2.csv already exists..."
    else
    curl -o trips/$1-citibike-tripdata.zip $3
    unzip -p trips/$1-citibike-tripdata.zip > trips/$2.csv
    rm trips/$1-citibike-tripdata.zip
    fi fi
}

for year in {2013..2020}
do
for month in {1..12}
do
for name in $year$month "${year}0${month}"
do
for end in ".csv.zip" ".zip"
do
url=https://s3.amazonaws.com/tripdata/$name-citibike-tripdata$end
filename="${year}_${month}"
download $name $filename $url
done
done
done
done
